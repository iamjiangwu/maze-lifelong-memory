"""Exhaustive, parameter-addressed 45x45 place-map caching for ED6.

The ED6 candidate table contains matched-neuron *pairs*, but many rows reuse a
session-neuron node.  This module reduces the table to unique nodes and builds
all maps for one session with a sparse bin-by-frame aggregation.  The cache
stores only the smoothed denoised-C map and its neuron-specific occupancy mask;
raw maps can be reconstructed for the much smaller final display set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage, sparse

from ca_analysis.spatial import discretize_positions


CACHE_SCHEMA_VERSION = "ed06-place-map-cache-v3-context-aware"
ALGORITHM_VERSION = "sparse-bin-by-frame-v1"
EXPECTED_CANDIDATE_ROWS = 17_841
EXPECTED_PAIR_COUNT = 36
EXPECTED_UNIQUE_NODES = 28_795
EXPECTED_CONTEXT_COUNT = 53
EXPECTED_SESSION_COUNT = 50

NEURAL_VARIANT_ORDINARY = "ordinary"
NEURAL_VARIANT_PRESORT = "presort"
NEURAL_VARIANT_SORTED = "sorted"
NEURAL_VARIANTS = frozenset(
    {NEURAL_VARIANT_ORDINARY, NEURAL_VARIANT_PRESORT, NEURAL_VARIANT_SORTED}
)

SOURCE_COMPONENT_VARIANTS = {
    "current_saved": NEURAL_VARIANT_ORDINARY,
    "new_12mo": NEURAL_VARIANT_ORDINARY,
    "restored_manual_exclusions_2026-07-12": NEURAL_VARIANT_SORTED,
    "corrected_842_2026-07-12": NEURAL_VARIANT_SORTED,
}

# The April 2026 current-saved 839 6->9 analysis used the 1,882-neuron
# pre-recreation day-9 coordinate system.  The June 2026 9->12 analysis uses
# the recreated 1,525-neuron ordinary file, so the biological session must be
# represented by two cache contexts.
PRESORT_SIDE_OVERRIDES = frozenset(
    {("current_saved", "839-6day1->839-9day1", "B")}
)


@dataclass(frozen=True)
class PlaceMapCacheParameters:
    """Scientific parameters that define one reusable ED6 map cache."""

    n_bins: int = 45
    sigma: float = 1.0
    dt: float = 0.04
    min_time_base: float = 0.5
    smoothing_mode: str = "constant"
    signal_source: str = "C"
    frame_column: str = "frame"
    position_columns: tuple[str, str] = ("mass_x", "mass_y")
    epsilon: float = 1e-12
    algorithm_version: str = ALGORITHM_VERSION

    @property
    def effective_min_time_s(self) -> float:
        return float(self.min_time_base) * (15.0 / float(self.n_bins)) ** 2

    def validate(self) -> None:
        if int(self.n_bins) <= 0:
            raise ValueError("n_bins must be positive")
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be nonnegative")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be positive")
        if float(self.min_time_base) < 0.0:
            raise ValueError("min_time_base must be nonnegative")
        if float(self.epsilon) < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if len(tuple(self.position_columns)) != 2:
            raise ValueError("position_columns must contain x and y")
        if str(self.signal_source) != "C":
            raise ValueError("the ED6 display cache must use denoised C")


@dataclass(frozen=True)
class SessionMapArrays:
    """Vectorized map results for one session, ordered by ``neuron_ids``."""

    neuron_ids: np.ndarray
    smooth_maps: np.ndarray
    occupancy_masks: np.ndarray
    n_selected_frames: int
    n_finite_position_frames: int
    n_dropped_position_frames: int


@dataclass(frozen=True)
class SessionInputPaths:
    """Minimal files needed for a session map, excluding trajectory data."""

    selected_csv: Path
    neural_mat: Path
    experiment_day: int
    neural_variant: str = NEURAL_VARIANT_ORDINARY
    context_key: str = ""


@dataclass(frozen=True, order=True)
class NeuralSessionContext:
    """One neuron-ID coordinate system for one biological session."""

    session_id: str
    neural_variant: str = NEURAL_VARIANT_ORDINARY

    def __post_init__(self) -> None:
        if not str(self.session_id):
            raise ValueError("session_id must be nonempty")
        if str(self.neural_variant) not in NEURAL_VARIANTS:
            raise ValueError(
                f"unsupported neural variant {self.neural_variant!r}; "
                f"expected one of {sorted(NEURAL_VARIANTS)}"
            )

    @property
    def context_key(self) -> str:
        return f"{self.session_id}::{self.neural_variant}"


def resolve_candidate_side_context(
    row: Mapping[str, Any] | pd.Series,
    side: str,
) -> NeuralSessionContext:
    """Resolve the exact neural coordinate system for one candidate side.

    ``side`` accepts ``A/src/train`` or ``B/dst/test``.  Resolution is based on
    immutable source-component provenance, not on whether a numeric neuron ID
    happens to fit inside a later neural matrix.
    """

    side_key = str(side).strip().lower()
    if side_key in {"a", "src", "source", "train"}:
        canonical_side = "A"
        session_column = "src_session"
    elif side_key in {"b", "dst", "destination", "test"}:
        canonical_side = "B"
        session_column = "dst_session"
    else:
        raise ValueError("side must be A/src/train or B/dst/test")
    required = {"source_component", "pair_key", session_column}
    missing = sorted(key for key in required if key not in row)
    if missing:
        raise ValueError(f"candidate row is missing context columns: {missing}")
    source_component = str(row["source_component"])
    pair_key = str(row["pair_key"])
    session_id = str(row[session_column])
    if not session_id:
        raise ValueError(f"{session_column} must be nonempty")
    try:
        neural_variant = SOURCE_COMPONENT_VARIANTS[source_component]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ED6 source component for neural context: {source_component!r}"
        ) from exc
    if (source_component, pair_key, canonical_side) in PRESORT_SIDE_OVERRIDES:
        neural_variant = NEURAL_VARIANT_PRESORT
    return NeuralSessionContext(session_id=session_id, neural_variant=neural_variant)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parameter_payload(parameters: PlaceMapCacheParameters) -> dict[str, Any]:
    """Return the full scientific cache contract as JSON-safe values."""

    parameters.validate()
    payload = asdict(parameters)
    payload["position_columns"] = list(parameters.position_columns)
    payload["effective_min_time_s"] = parameters.effective_min_time_s
    return payload


def parameter_hash(parameters: PlaceMapCacheParameters) -> str:
    """SHA-256 hash of every parameter that can change the cached map."""

    return hashlib.sha256(_canonical_json(parameter_payload(parameters)).encode("utf-8")).hexdigest()


def neuron_id_hash(neuron_ids: Sequence[int] | np.ndarray) -> str:
    ids = _sorted_unique_neuron_ids(neuron_ids)
    return hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _sorted_unique_neuron_ids(values: Sequence[int] | np.ndarray) -> np.ndarray:
    numeric = np.asarray(list(values), dtype=float).ravel()
    if numeric.size == 0:
        raise ValueError("at least one neuron ID is required")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("neuron IDs must be finite integers")
    ids = np.unique(numeric.astype(np.int64))
    if int(ids.min()) < 0:
        raise ValueError("neuron IDs must be nonnegative")
    return ids


def build_context_neuron_index(
    pool: pd.DataFrame,
    *,
    strict_ed6: bool = True,
) -> pd.DataFrame:
    """Reduce ED6 rows to unique context-aware neuron nodes.

    Numeric neuron IDs are meaningful only within a neural coordinate system.
    In particular, ``839-9day1::ordinary`` and ``839-9day1::presort`` are
    deliberately distinct contexts even where their numeric IDs overlap.
    """

    required = {
        "source_component",
        "pair_key",
        "src_session",
        "dst_session",
        "train_id",
        "test_id",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"candidate pool is missing columns: {missing}")
    if strict_ed6:
        if len(pool) != EXPECTED_CANDIDATE_ROWS:
            raise AssertionError(
                f"expected {EXPECTED_CANDIDATE_ROWS:,} ED6 rows, found {len(pool):,}"
            )
        if int(pool["pair_key"].astype(str).nunique()) != EXPECTED_PAIR_COUNT:
            raise AssertionError(
                f"expected {EXPECTED_PAIR_COUNT} ED6 pairs, found "
                f"{pool['pair_key'].astype(str).nunique()}"
            )

    def side(session_col: str, neuron_col: str, role: str) -> pd.DataFrame:
        neuron = pd.to_numeric(pool[neuron_col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(neuron).all() or not np.equal(neuron, np.floor(neuron)).all():
            raise ValueError(f"{neuron_col} must contain finite integer IDs")
        contexts = [
            resolve_candidate_side_context(row, role)
            for row in pool.to_dict(orient="records")
        ]
        return pd.DataFrame(
            {
                "session_id": [context.session_id for context in contexts],
                "neural_variant": [context.neural_variant for context in contexts],
                "context_key": [context.context_key for context in contexts],
                "neuron_id": neuron.astype(np.int64),
                "role": role,
                "pair_key": pool["pair_key"].astype(str).to_numpy(),
                "source_component": pool["source_component"].astype(str).to_numpy(),
            }
        )

    long = pd.concat(
        [side("src_session", "train_id", "A"), side("dst_session", "test_id", "B")],
        ignore_index=True,
    )
    if (long["neuron_id"] < 0).any() or long["session_id"].eq("").any():
        raise ValueError("session IDs must be nonempty and neuron IDs nonnegative")
    grouped = (
        long.groupby(
            ["context_key", "session_id", "neural_variant", "neuron_id"],
            sort=True,
            observed=True,
        )
        .agg(
            roles=("role", lambda values: ",".join(sorted(set(map(str, values))))),
            n_candidate_uses=("pair_key", "size"),
            n_pair_uses=("pair_key", "nunique"),
            source_components=(
                "source_component",
                lambda values: ",".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
        .sort_values(["context_key", "neuron_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if strict_ed6:
        if len(grouped) != EXPECTED_UNIQUE_NODES:
            raise AssertionError(
                f"expected {EXPECTED_UNIQUE_NODES:,} unique context-neuron nodes, "
                f"found {len(grouped):,}"
            )
        if int(grouped["context_key"].nunique()) != EXPECTED_CONTEXT_COUNT:
            raise AssertionError(
                f"expected {EXPECTED_CONTEXT_COUNT} neural contexts, found "
                f"{grouped['context_key'].nunique()}"
            )
        if int(grouped["session_id"].nunique()) != EXPECTED_SESSION_COUNT:
            raise AssertionError(
                f"expected {EXPECTED_SESSION_COUNT} biological sessions, found "
                f"{grouped['session_id'].nunique()}"
            )
    return grouped


def build_session_neuron_index(
    pool: pd.DataFrame,
    *,
    strict_ed6: bool = True,
) -> pd.DataFrame:
    """Backward-named alias for the context-aware node index."""

    return build_context_neuron_index(pool, strict_ed6=strict_ed6)


def collect_context_neuron_needs(
    pool: pd.DataFrame,
    *,
    strict_ed6: bool = True,
) -> dict[NeuralSessionContext, np.ndarray]:
    """Collect sorted neuron IDs for each distinct neural session context."""

    index = build_context_neuron_index(pool, strict_ed6=strict_ed6)
    return {
        NeuralSessionContext(str(session_id), str(neural_variant)): group[
            "neuron_id"
        ].to_numpy(dtype=np.int64)
        for (session_id, neural_variant), group in index.groupby(
            ["session_id", "neural_variant"], sort=True, observed=True
        )
    }


def collect_session_neuron_needs(
    pool: pd.DataFrame,
    *,
    strict_ed6: bool = True,
) -> dict[NeuralSessionContext, np.ndarray]:
    """Backward-named alias that intentionally returns context-aware keys."""

    return collect_context_neuron_needs(pool, strict_ed6=strict_ed6)


def _selected_frames_and_positions(
    selected_df: pd.DataFrame,
    c_matrix_or_shape: Any,
    parameters: PlaceMapCacheParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {parameters.frame_column, *parameters.position_columns}
    missing = sorted(required - set(selected_df.columns))
    if missing:
        raise ValueError(f"selected_df is missing columns: {missing}")
    frame_values = pd.to_numeric(selected_df[parameters.frame_column], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(frame_values).all() or not np.equal(frame_values, np.floor(frame_values)).all():
        raise ValueError("selected frame indices must be finite integers")
    frames = frame_values.astype(np.int64)
    c_shape = tuple(getattr(c_matrix_or_shape, "shape", c_matrix_or_shape))
    if len(c_shape) != 2:
        raise ValueError("C shape must be (neurons, frames)")
    if frames.size and (int(frames.min()) < 0 or int(frames.max()) >= int(c_shape[1])):
        raise IndexError("selected frames are out of C bounds")
    positions = selected_df.loc[:, list(parameters.position_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    finite_positions = np.isfinite(positions).all(axis=1)
    return frames, positions, finite_positions


def _compute_maps_with_count_loader(
    selected_df: pd.DataFrame,
    neuron_ids: Sequence[int] | np.ndarray,
    *,
    c_shape: tuple[int, int],
    count_loader: Callable[[np.ndarray, np.ndarray], np.ndarray],
    parameters: PlaceMapCacheParameters,
    chunk_size: int,
    output_dtype: Any,
) -> SessionMapArrays:
    """Shared sparse aggregation engine for in-memory and selective-HDF5 C."""

    params = parameters
    params.validate()
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    ids = _sorted_unique_neuron_ids(neuron_ids)
    if int(ids.max()) >= int(c_shape[0]):
        raise IndexError(f"neuron {int(ids.max())} is out of C bounds {int(c_shape[0])}")
    frames, positions, finite_positions = _selected_frames_and_positions(
        selected_df.reset_index(drop=True), c_shape, params
    )
    valid_frames = frames[finite_positions]
    valid_positions = positions[finite_positions]
    n_flat = int(params.n_bins) ** 2
    if valid_positions.shape[0]:
        labels, _, _ = discretize_positions(valid_positions, int(params.n_bins), int(params.n_bins))
        labels = np.asarray(labels, dtype=np.int64)
        bin_by_frame = sparse.csr_matrix(
            (
                np.ones(labels.size, dtype=np.float64),
                (labels, np.arange(labels.size, dtype=np.int64)),
            ),
            shape=(n_flat, labels.size),
            dtype=np.float64,
        )
    else:
        bin_by_frame = sparse.csr_matrix((n_flat, 0), dtype=np.float64)

    output_type = np.dtype(output_dtype)
    if output_type.kind != "f":
        raise ValueError("output_dtype must be floating point")
    maps = np.empty((len(ids), int(params.n_bins), int(params.n_bins)), dtype=output_type)
    masks = np.empty(maps.shape, dtype=bool)
    min_time = float(params.effective_min_time_s)
    for start in range(0, len(ids), int(chunk_size)):
        stop = min(start + int(chunk_size), len(ids))
        block_ids = ids[start:stop]
        counts = np.asarray(count_loader(block_ids, valid_frames), dtype=np.float64)
        expected_count_shape = (len(block_ids), len(valid_frames))
        if tuple(counts.shape) != expected_count_shape:
            raise ValueError(
                f"count loader returned {counts.shape}, expected {expected_count_shape}"
            )
        finite_counts = np.isfinite(counts)
        safe_counts = np.where(finite_counts, counts, 0.0)
        count_sums = np.asarray(bin_by_frame @ safe_counts.T, dtype=np.float64).T
        occupancy = (
            np.asarray(bin_by_frame @ finite_counts.astype(np.float64).T, dtype=np.float64).T
            * float(params.dt)
        )
        occupied = occupancy >= min_time
        with np.errstate(divide="ignore", invalid="ignore"):
            tuning = count_sums / np.maximum(occupancy, 1e-12)
        tuning[~occupied] = 0.0
        tuning = tuning.reshape(len(block_ids), int(params.n_bins), int(params.n_bins))
        if float(params.sigma) > 0.0:
            tuning = ndimage.gaussian_filter(
                tuning,
                sigma=(0.0, float(params.sigma), float(params.sigma)),
                mode=str(params.smoothing_mode),
            )
        tuning = tuning + float(params.epsilon)
        maps[start:stop] = tuning.astype(output_type, copy=False)
        masks[start:stop] = occupied.reshape(
            len(block_ids), int(params.n_bins), int(params.n_bins)
        )
    return SessionMapArrays(
        neuron_ids=ids,
        smooth_maps=maps,
        occupancy_masks=masks,
        n_selected_frames=int(len(frames)),
        n_finite_position_frames=int(finite_positions.sum()),
        n_dropped_position_frames=int((~finite_positions).sum()),
    )


def compute_session_maps_vectorized(
    c_matrix: np.ndarray,
    selected_df: pd.DataFrame,
    neuron_ids: Sequence[int] | np.ndarray,
    *,
    parameters: PlaceMapCacheParameters | None = None,
    chunk_size: int = 64,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> SessionMapArrays:
    """Compute one session's neuron maps with sparse bin-by-frame products.

    Position labels are computed once.  Within each neuron block, nonfinite C
    samples are excluded independently, preserving the scalar implementation's
    neuron-specific occupancy semantics.
    """

    params = parameters or PlaceMapCacheParameters()
    params.validate()
    if getattr(c_matrix, "ndim", None) != 2:
        raise ValueError("c_matrix must have shape (neurons, frames)")
    return _compute_maps_with_count_loader(
        selected_df,
        neuron_ids,
        c_shape=(int(c_matrix.shape[0]), int(c_matrix.shape[1])),
        count_loader=lambda block_ids, frames: (
            np.asarray(c_matrix[block_ids][:, frames], dtype=np.float64)
            if frames.size
            else np.empty((len(block_ids), 0), dtype=np.float64)
        ),
        parameters=params,
        chunk_size=int(chunk_size),
        output_dtype=output_dtype,
    )


def _resolve_neural_mat_under_root(session_id: str, neural_root: Path) -> tuple[Path, int]:
    """Resolve a maintained neural MAT and experiment day under one root."""

    from ca_analysis.io import _extract_info

    parsed = _extract_info(str(session_id))
    if parsed is None:
        raise ValueError(f"could not parse session nickname: {session_id}")
    mouse_id, experiment_type, experiment_day = parsed
    directory_names = []
    if int(experiment_type) != 0:
        directory_names.append(f"{mouse_id}-{int(experiment_type)}")
    directory_names.append(str(mouse_id))
    filename_candidates = (
        f"final_day{int(experiment_day)}.mat",
        f"{int(experiment_day)}.mat",
        f"day{int(experiment_day)}.mat",
        f"final_{int(experiment_day)}.mat",
    )
    neural_mat = next(
        (
            neural_root / directory / filename
            for directory in directory_names
            for filename in filename_candidates
            if (neural_root / directory / filename).exists()
        ),
        None,
    )
    if neural_mat is None:
        raise FileNotFoundError(
            f"no maintained neural MAT naming pattern found for session {session_id} under {neural_root}"
        )
    return neural_mat, int(experiment_day)


def resolve_context_input_paths(
    context: NeuralSessionContext,
    *,
    behav_wd: Path | str,
    neural_wd: Path | str,
    sorted_neural_wd: Path | str | None = None,
) -> SessionInputPaths:
    """Resolve behavior plus the exact ordinary/presort/sorted neural MAT."""

    session_id = str(context.session_id)
    selected_csv = Path(behav_wd) / "selected_data" / f"{session_id}_selected_data.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"missing selected behavior CSV: {selected_csv}")
    if context.neural_variant == NEURAL_VARIANT_SORTED:
        if sorted_neural_wd is None:
            raise ValueError("sorted_neural_wd is required for a sorted neural context")
        neural_mat, experiment_day = _resolve_neural_mat_under_root(
            session_id, Path(sorted_neural_wd)
        )
    else:
        ordinary_mat, experiment_day = _resolve_neural_mat_under_root(
            session_id, Path(neural_wd)
        )
        neural_mat = ordinary_mat
    if context.neural_variant == NEURAL_VARIANT_PRESORT:
        neural_mat = ordinary_mat.parent / "presort" / ordinary_mat.name
    if not neural_mat.exists():
        raise FileNotFoundError(
            f"missing {context.neural_variant} neural MAT for {session_id}: {neural_mat}"
        )
    return SessionInputPaths(
        selected_csv=selected_csv,
        neural_mat=neural_mat,
        experiment_day=int(experiment_day),
        neural_variant=context.neural_variant,
        context_key=context.context_key,
    )


def resolve_session_input_paths(
    session_id: str,
    *,
    behav_wd: Path | str,
    neural_wd: Path | str,
    neural_variant: str = NEURAL_VARIANT_ORDINARY,
    sorted_neural_wd: Path | str | None = None,
) -> SessionInputPaths:
    """Compatibility wrapper around :func:`resolve_context_input_paths`."""

    return resolve_context_input_paths(
        NeuralSessionContext(str(session_id), str(neural_variant)),
        behav_wd=behav_wd,
        neural_wd=neural_wd,
        sorted_neural_wd=sorted_neural_wd,
    )


def _merged_frame_ranges(unique_frames: np.ndarray, *, max_gap_frames: int) -> list[tuple[int, int]]:
    """Return half-open read ranges, merging short gaps to reduce HDF5 calls."""

    frames = np.asarray(unique_frames, dtype=np.int64)
    if frames.size == 0:
        return []
    split_at = np.flatnonzero(np.diff(frames) > int(max_gap_frames) + 1) + 1
    groups = np.split(frames, split_at)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def read_selected_c_block(
    dataset: h5py.Dataset,
    neuron_ids: Sequence[int] | np.ndarray,
    frames: Sequence[int] | np.ndarray,
    *,
    max_gap_frames: int = 64,
) -> np.ndarray:
    """Read only requested C values from MATLAB's frames-by-neurons dataset."""

    ids = _sorted_unique_neuron_ids(neuron_ids)
    frame_array = np.asarray(frames, dtype=np.int64).ravel()
    if frame_array.size == 0:
        return np.empty((len(ids), 0), dtype=np.float64)
    if int(frame_array.min()) < 0 or int(frame_array.max()) >= int(dataset.shape[0]):
        raise IndexError("requested selected frames are out of stored C bounds")
    if int(ids.max()) >= int(dataset.shape[1]):
        raise IndexError("requested neuron IDs are out of stored C bounds")
    if int(max_gap_frames) < 0:
        raise ValueError("max_gap_frames must be nonnegative")
    unique_frames, inverse = np.unique(frame_array, return_inverse=True)
    unique_values = np.empty((len(unique_frames), len(ids)), dtype=np.float64)
    for range_start, range_stop in _merged_frame_ranges(
        unique_frames, max_gap_frames=int(max_gap_frames)
    ):
        in_range = (unique_frames >= range_start) & (unique_frames < range_stop)
        offsets = unique_frames[in_range] - int(range_start)
        # h5py permits one fancy axis: frames use a slice and neurons use the
        # sorted ID vector.  Short gaps are read and discarded to bound calls.
        slab = np.asarray(dataset[range_start:range_stop, ids], dtype=np.float64)
        unique_values[in_range] = slab[offsets]
    return unique_values[inverse].T


def compute_session_maps_from_files(
    session_id: str,
    neuron_ids: Sequence[int] | np.ndarray,
    *,
    behav_wd: Path | str,
    neural_wd: Path | str,
    neural_variant: str = NEURAL_VARIANT_ORDINARY,
    sorted_neural_wd: Path | str | None = None,
    parameters: PlaceMapCacheParameters | None = None,
    chunk_size: int = 64,
    max_gap_frames: int = 64,
    time_block_frames: int = 4096,
    output_dtype: Any = np.float32,
) -> tuple[SessionMapArrays, dict[str, Any]]:
    """Build maps in one time-block pass over C only.

    MATLAB C datasets in this cohort are chunked as a few frames by all
    neurons.  Reading one neuron block at a time would therefore decompress the
    same HDF5 chunks repeatedly.  This path streams each needed time block once,
    selects requested neurons in NumPy, and immediately accumulates sparse
    bin-by-frame sums.
    """

    params = parameters or PlaceMapCacheParameters()
    if int(time_block_frames) <= 0:
        raise ValueError("time_block_frames must be positive")
    context = NeuralSessionContext(str(session_id), str(neural_variant))
    paths = resolve_context_input_paths(
        context,
        behav_wd=behav_wd,
        neural_wd=neural_wd,
        sorted_neural_wd=sorted_neural_wd,
    )
    selected_df = pd.read_csv(
        paths.selected_csv,
        usecols=[params.frame_column, *params.position_columns],
    )
    with h5py.File(paths.neural_mat, "r") as handle:
        if "C" not in handle or not isinstance(handle["C"], h5py.Dataset):
            raise RuntimeError(f"session {session_id} MAT has no dense C dataset")
        dataset = handle["C"]
        if dataset.ndim != 2:
            raise ValueError(f"session {session_id} C dataset must be two-dimensional")
        # Maintained load_behav_neural applies .T: MATLAB stores frames x neurons.
        stored_shape = (int(dataset.shape[0]), int(dataset.shape[1]))
        ids = _sorted_unique_neuron_ids(neuron_ids)
        if int(ids.max()) >= stored_shape[1]:
            raise IndexError(
                f"session {session_id} neuron {int(ids.max())} is out of stored C bounds "
                f"{stored_shape[1]}"
            )
        frames, positions, finite_positions = _selected_frames_and_positions(
            selected_df.reset_index(drop=True), (stored_shape[1], stored_shape[0]), params
        )
        valid_frames = frames[finite_positions]
        valid_positions = positions[finite_positions]
        n_flat = int(params.n_bins) ** 2
        count_sums = np.zeros((n_flat, len(ids)), dtype=np.float64)
        occupancy_samples = np.zeros_like(count_sums)
        read_block_count = 0
        read_frame_count = 0
        if len(valid_frames):
            labels, _, _ = discretize_positions(
                valid_positions, int(params.n_bins), int(params.n_bins)
            )
            labels = np.asarray(labels, dtype=np.int64)
            first_block = (int(valid_frames.min()) // int(time_block_frames)) * int(
                time_block_frames
            )
            final_frame = int(valid_frames.max()) + 1
            for block_start in range(first_block, final_frame, int(time_block_frames)):
                block_stop = min(block_start + int(time_block_frames), stored_shape[0])
                in_block = (valid_frames >= block_start) & (valid_frames < block_stop)
                if not np.any(in_block):
                    continue
                # Read all columns once because the source HDF5 chunks span all
                # neurons; select the ED6 node subset only after decompression.
                slab = np.asarray(dataset[block_start:block_stop, :], dtype=np.float64)
                offsets = valid_frames[in_block] - int(block_start)
                block_counts = slab[offsets][:, ids]
                finite_counts = np.isfinite(block_counts)
                safe_counts = np.where(finite_counts, block_counts, 0.0)
                block_labels = labels[in_block]
                bin_by_frame = sparse.csr_matrix(
                    (
                        np.ones(len(block_labels), dtype=np.float64),
                        (block_labels, np.arange(len(block_labels), dtype=np.int64)),
                    ),
                    shape=(n_flat, len(block_labels)),
                    dtype=np.float64,
                )
                count_sums += np.asarray(bin_by_frame @ safe_counts, dtype=np.float64)
                occupancy_samples += np.asarray(
                    bin_by_frame @ finite_counts.astype(np.float64), dtype=np.float64
                )
                read_block_count += 1
                read_frame_count += int(block_stop - block_start)
        occupancy = occupancy_samples.T * float(params.dt)
        occupied = occupancy >= float(params.effective_min_time_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            tuning = count_sums.T / np.maximum(occupancy, 1e-12)
        tuning[~occupied] = 0.0
        tuning = tuning.reshape(len(ids), int(params.n_bins), int(params.n_bins))
        if float(params.sigma) > 0.0:
            tuning = ndimage.gaussian_filter(
                tuning,
                sigma=(0.0, float(params.sigma), float(params.sigma)),
                mode=str(params.smoothing_mode),
            )
        tuning = tuning + float(params.epsilon)
        arrays = SessionMapArrays(
            neuron_ids=ids,
            smooth_maps=tuning.astype(np.dtype(output_dtype), copy=False),
            occupancy_masks=occupied.reshape(
                len(ids), int(params.n_bins), int(params.n_bins)
            ),
            n_selected_frames=int(len(frames)),
            n_finite_position_frames=int(finite_positions.sum()),
            n_dropped_position_frames=int((~finite_positions).sum()),
        )
        hdf5_chunk_shape = list(map(int, dataset.chunks)) if dataset.chunks is not None else None
        hdf5_compression = str(dataset.compression) if dataset.compression is not None else None
    unique_frames = np.unique(valid_frames)
    metadata = {
        "session_id": str(session_id),
        "context_key": context.context_key,
        "neural_variant": context.neural_variant,
        "selected_csv": str(paths.selected_csv),
        "neural_mat": str(paths.neural_mat),
        "selected_csv_size_bytes": int(paths.selected_csv.stat().st_size),
        "selected_csv_mtime_ns": int(paths.selected_csv.stat().st_mtime_ns),
        "neural_mat_size_bytes": int(paths.neural_mat.stat().st_size),
        "neural_mat_mtime_ns": int(paths.neural_mat.stat().st_mtime_ns),
        "stored_c_shape_frames_by_neurons": list(stored_shape),
        "canonical_c_shape_neurons_by_frames": [stored_shape[1], stored_shape[0]],
        "hdf5_orientation": "frames_by_neurons; transposed by maintained loader",
        "hdf5_datasets_read": ["C"],
        "selected_csv_columns_read": [params.frame_column, *params.position_columns],
        "n_unique_selected_frames": int(len(unique_frames)),
        "hdf5_chunk_shape": hdf5_chunk_shape,
        "hdf5_compression": hdf5_compression,
        "hdf5_time_block_frames": int(time_block_frames),
        "hdf5_time_blocks_read": int(read_block_count),
        "hdf5_frames_read_total": read_frame_count,
        "hdf5_frame_read_amplification": (
            float(read_frame_count / len(unique_frames)) if len(unique_frames) else 0.0
        ),
        "hdf5_passes_over_c": 1,
        "max_gap_frames": int(max_gap_frames),
        "legacy_neuron_chunk_size_unused_for_hdf5": int(chunk_size),
    }
    return arrays, metadata


def compute_context_maps_from_files(
    context: NeuralSessionContext,
    neuron_ids: Sequence[int] | np.ndarray,
    *,
    behav_wd: Path | str,
    neural_wd: Path | str,
    sorted_neural_wd: Path | str | None,
    parameters: PlaceMapCacheParameters | None = None,
    chunk_size: int = 64,
    max_gap_frames: int = 64,
    time_block_frames: int = 4096,
    output_dtype: Any = np.float32,
) -> tuple[SessionMapArrays, dict[str, Any]]:
    """Compute one explicitly resolved neural-context cache payload."""

    return compute_session_maps_from_files(
        context.session_id,
        neuron_ids,
        behav_wd=behav_wd,
        neural_wd=neural_wd,
        neural_variant=context.neural_variant,
        sorted_neural_wd=sorted_neural_wd,
        parameters=parameters,
        chunk_size=chunk_size,
        max_gap_frames=max_gap_frames,
        time_block_frames=time_block_frames,
        output_dtype=output_dtype,
    )


def session_cache_paths(
    cache_dir: Path | str,
    session_id: str,
    neuron_ids: Sequence[int] | np.ndarray,
    parameters: PlaceMapCacheParameters,
    *,
    neural_variant: str = NEURAL_VARIANT_ORDINARY,
) -> tuple[Path, Path]:
    context = NeuralSessionContext(str(session_id), str(neural_variant))
    directory = Path(cache_dir)
    stem = (
        f"v3__{_safe_name(context.context_key)}__p{parameter_hash(parameters)[:16]}"
        f"__n{neuron_id_hash(neuron_ids)[:16]}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.manifest.json"


def context_cache_paths(
    cache_dir: Path | str,
    context: NeuralSessionContext,
    neuron_ids: Sequence[int] | np.ndarray,
    parameters: PlaceMapCacheParameters,
) -> tuple[Path, Path]:
    """Return versioned cache paths for one explicit neural context."""

    return session_cache_paths(
        cache_dir,
        context.session_id,
        neuron_ids,
        parameters,
        neural_variant=context.neural_variant,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_session_map_cache(
    cache_dir: Path | str,
    session_id: str,
    arrays: SessionMapArrays,
    *,
    parameters: PlaceMapCacheParameters | None = None,
    c_shape: Sequence[int] | None = None,
    chunk_size: int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    neural_variant: str = NEURAL_VARIANT_ORDINARY,
) -> dict[str, Any]:
    """Atomically write one session NPZ and its auditable JSON manifest."""

    params = parameters or PlaceMapCacheParameters()
    params.validate()
    ids = _sorted_unique_neuron_ids(arrays.neuron_ids)
    expected_shape = (len(ids), int(params.n_bins), int(params.n_bins))
    if tuple(arrays.smooth_maps.shape) != expected_shape:
        raise ValueError(f"smooth_maps has shape {arrays.smooth_maps.shape}, expected {expected_shape}")
    if tuple(arrays.occupancy_masks.shape) != expected_shape:
        raise ValueError(
            f"occupancy_masks has shape {arrays.occupancy_masks.shape}, expected {expected_shape}"
        )
    if not np.array_equal(np.asarray(arrays.neuron_ids, dtype=np.int64), ids):
        raise ValueError("SessionMapArrays.neuron_ids must be sorted and unique")
    context = NeuralSessionContext(str(session_id), str(neural_variant))
    cache_path, manifest_path = context_cache_paths(cache_dir, context, ids, params)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache = cache_path.with_name(f"{cache_path.name}.tmp.npz")
    np.savez_compressed(
        tmp_cache,
        neuron_ids=ids.astype(np.int64, copy=False),
        smooth_maps=np.asarray(arrays.smooth_maps, dtype=np.float32),
        occupancy_masks=np.asarray(arrays.occupancy_masks, dtype=bool),
    )
    tmp_cache.replace(cache_path)
    manifest: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "session_id": str(session_id),
        "context_key": context.context_key,
        "neural_variant": context.neural_variant,
        "parameter_hash": parameter_hash(params),
        "parameters": parameter_payload(params),
        "neuron_id_hash": neuron_id_hash(ids),
        "neuron_ids": ids.tolist(),
        "n_neurons": int(len(ids)),
        "n_selected_frames": int(arrays.n_selected_frames),
        "n_finite_position_frames": int(arrays.n_finite_position_frames),
        "n_dropped_position_frames": int(arrays.n_dropped_position_frames),
        "c_shape": list(map(int, c_shape)) if c_shape is not None else None,
        "chunk_size": int(chunk_size) if chunk_size is not None else None,
        "source_metadata": dict(source_metadata) if source_metadata is not None else None,
        "array_shapes": {
            "smooth_maps": list(map(int, expected_shape)),
            "occupancy_masks": list(map(int, expected_shape)),
        },
        "array_dtypes": {"smooth_maps": "float32", "occupancy_masks": "bool"},
        "cache_file": cache_path.name,
        "cache_file_sha256": _sha256_file(cache_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    return manifest


def load_session_map_cache(
    cache_path: Path | str,
    manifest_path: Path | str,
    *,
    expected_session_id: str | None = None,
    expected_neural_variant: str | None = None,
    expected_context: NeuralSessionContext | None = None,
    expected_neuron_ids: Sequence[int] | np.ndarray | None = None,
    expected_parameters: PlaceMapCacheParameters | None = None,
    verify_file_hash: bool = True,
) -> tuple[SessionMapArrays, dict[str, Any]]:
    """Load and validate a session cache against its manifest contract."""

    cache = Path(cache_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported ED6 cache schema")
    if expected_session_id is not None and str(manifest.get("session_id")) != str(
        expected_session_id
    ):
        raise ValueError("cache session ID does not match expectation")
    if expected_context is not None:
        if str(manifest.get("context_key")) != expected_context.context_key:
            raise ValueError("cache neural context does not match expectation")
    if expected_neural_variant is not None and str(manifest.get("neural_variant")) != str(
        expected_neural_variant
    ):
        raise ValueError("cache neural variant does not match expectation")
    if expected_parameters is not None and manifest.get("parameter_hash") != parameter_hash(
        expected_parameters
    ):
        raise ValueError("cache parameter hash does not match expectation")
    if verify_file_hash and str(manifest.get("cache_file_sha256")) != _sha256_file(cache):
        raise ValueError("cache file SHA-256 does not match its manifest")
    with np.load(cache, allow_pickle=False) as payload:
        ids = np.asarray(payload["neuron_ids"], dtype=np.int64)
        maps = np.asarray(payload["smooth_maps"], dtype=np.float32)
        masks = np.asarray(payload["occupancy_masks"], dtype=bool)
    if expected_neuron_ids is not None:
        expected_ids = _sorted_unique_neuron_ids(expected_neuron_ids)
        if not np.array_equal(ids, expected_ids):
            raise ValueError("cached neuron IDs do not match expectation")
    if manifest.get("neuron_id_hash") != neuron_id_hash(ids):
        raise ValueError("cached neuron IDs do not match the manifest hash")
    expected_shape = (len(ids), int(maps.shape[1]), int(maps.shape[2]))
    if maps.ndim != 3 or masks.shape != maps.shape or maps.shape != expected_shape:
        raise ValueError("cached array shapes are inconsistent")
    manifest_shape = tuple(manifest.get("array_shapes", {}).get("smooth_maps", ()))
    if tuple(maps.shape) != manifest_shape:
        raise ValueError("cached map shape does not match manifest")
    arrays = SessionMapArrays(
        neuron_ids=ids,
        smooth_maps=maps,
        occupancy_masks=masks,
        n_selected_frames=int(manifest["n_selected_frames"]),
        n_finite_position_frames=int(manifest["n_finite_position_frames"]),
        n_dropped_position_frames=int(manifest["n_dropped_position_frames"]),
    )
    return arrays, manifest


def load_context_map_cache(
    cache_dir: Path | str,
    context: NeuralSessionContext,
    neuron_ids: Sequence[int] | np.ndarray,
    parameters: PlaceMapCacheParameters,
    *,
    verify_file_hash: bool = True,
) -> tuple[SessionMapArrays, dict[str, Any]]:
    """Resolve, load, and validate the cache for one explicit context."""

    cache_path, manifest_path = context_cache_paths(
        cache_dir, context, neuron_ids, parameters
    )
    return load_session_map_cache(
        cache_path,
        manifest_path,
        expected_session_id=context.session_id,
        expected_neural_variant=context.neural_variant,
        expected_context=context,
        expected_neuron_ids=neuron_ids,
        expected_parameters=parameters,
        verify_file_hash=verify_file_hash,
    )


def build_exhaustive_session_cache(
    pool: pd.DataFrame,
    *,
    cache_dir: Path | str,
    behav_wd: str,
    neural_wd: str,
    sorted_neural_wd: str | None = None,
    parameters: PlaceMapCacheParameters | None = None,
    chunk_size: int = 64,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = True,
    strict_ed6: bool = True,
    session_loader: Callable[..., tuple[Any, ...]] | None = None,
    max_gap_frames: int = 64,
    time_block_frames: int = 4096,
) -> pd.DataFrame:
    """Build all expected cache files for one deterministic session shard."""

    params = parameters or PlaceMapCacheParameters()
    params.validate()
    if int(shard_count) < 1 or not (0 <= int(shard_index) < int(shard_count)):
        raise ValueError("shard_index must be in [0, shard_count), with shard_count positive")
    needs = collect_context_neuron_needs(pool, strict_ed6=strict_ed6)
    all_contexts = sorted(needs)
    selected_contexts = all_contexts[int(shard_index) :: int(shard_count)]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for position, context in enumerate(selected_contexts, start=1):
        session_id = context.session_id
        ids = needs[context]
        cache_path, manifest_path = context_cache_paths(directory, context, ids, params)
        status = "computed"
        if bool(resume) and cache_path.exists() and manifest_path.exists():
            arrays, manifest = load_session_map_cache(
                cache_path,
                manifest_path,
                expected_session_id=session_id,
                expected_context=context,
                expected_neuron_ids=ids,
                expected_parameters=params,
            )
            status = "reused"
        else:
            if session_loader is None:
                arrays, source_metadata = compute_context_maps_from_files(
                    context,
                    ids,
                    behav_wd=behav_wd,
                    neural_wd=neural_wd,
                    sorted_neural_wd=sorted_neural_wd,
                    parameters=params,
                    chunk_size=int(chunk_size),
                    max_gap_frames=int(max_gap_frames),
                    time_block_frames=int(time_block_frames),
                    output_dtype=np.float32,
                )
                c_shape = source_metadata["canonical_c_shape_neurons_by_frames"]
            else:
                if context.neural_variant == NEURAL_VARIANT_PRESORT:
                    raise ValueError(
                        "custom session_loader cannot safely resolve a presort context"
                    )
                effective_neural_wd = (
                    sorted_neural_wd
                    if context.neural_variant == NEURAL_VARIANT_SORTED
                    else neural_wd
                )
                if effective_neural_wd is None:
                    raise ValueError("sorted_neural_wd is required for a sorted context")
                loaded = session_loader(
                    session_id, behav_wd=behav_wd, neural_wd=effective_neural_wd
                )
                if len(loaded) < 5:
                    raise RuntimeError(f"session loader returned an incomplete bundle for {session_id}")
                selected_df = loaded[1]
                c_matrix = loaded[4]
                if c_matrix is None:
                    raise RuntimeError(f"session {session_id} has no denoised C matrix")
                arrays = compute_session_maps_vectorized(
                    c_matrix,
                    selected_df,
                    ids,
                    parameters=params,
                    chunk_size=int(chunk_size),
                    output_dtype=np.float32,
                )
                c_shape = c_matrix.shape
                source_metadata = {
                    "session_id": str(session_id),
                    "context_key": context.context_key,
                    "neural_variant": context.neural_variant,
                    "loader": getattr(session_loader, "__name__", type(session_loader).__name__),
                    "hdf5_datasets_read": None,
                }
            manifest = write_session_map_cache(
                directory,
                session_id,
                arrays,
                parameters=params,
                c_shape=c_shape,
                chunk_size=int(chunk_size),
                source_metadata=source_metadata,
                neural_variant=context.neural_variant,
            )
        rows.append(
            {
                "session_order_in_shard": int(position),
                "session_id": session_id,
                "context_key": context.context_key,
                "neural_variant": context.neural_variant,
                "n_neurons": int(len(arrays.neuron_ids)),
                "n_selected_frames": int(arrays.n_selected_frames),
                "n_finite_position_frames": int(arrays.n_finite_position_frames),
                "n_dropped_position_frames": int(arrays.n_dropped_position_frames),
                "cache_status": status,
                "cache_file": cache_path.name,
                "manifest_file": manifest_path.name,
                "parameter_hash": manifest["parameter_hash"],
                "neuron_id_hash": manifest["neuron_id_hash"],
                "cache_file_sha256": manifest["cache_file_sha256"],
            }
        )
        print(
            f"[{position}/{len(selected_contexts)}; shard {int(shard_index) + 1}/{int(shard_count)}] "
            f"{context.context_key}: {len(ids):,} neurons ({status})",
            flush=True,
        )
    index = pd.DataFrame(rows)
    shard_stem = f"cache_index_shard_{int(shard_index):03d}_of_{int(shard_count):03d}"
    index.to_csv(directory / f"{shard_stem}.csv", index=False)
    run_manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "parameter_hash": parameter_hash(params),
        "parameters": parameter_payload(params),
        "strict_ed6": bool(strict_ed6),
        "expected_candidate_rows": int(len(pool)),
        "expected_unique_nodes": int(sum(len(values) for values in needs.values())),
        "expected_contexts": int(len(needs)),
        "expected_biological_sessions": int(len({item.session_id for item in needs})),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "selected_contexts": [item.context_key for item in selected_contexts],
        "neural_wd": str(Path(neural_wd)),
        "sorted_neural_wd": (
            str(Path(sorted_neural_wd)) if sorted_neural_wd is not None else None
        ),
        "chunk_size": int(chunk_size),
        "max_gap_frames": int(max_gap_frames),
        "time_block_frames": int(time_block_frames),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (directory / f"{shard_stem}.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def assemble_exhaustive_cache_index(
    pool: pd.DataFrame,
    *,
    cache_dir: Path | str,
    parameters: PlaceMapCacheParameters | None = None,
    strict_ed6: bool = True,
    verify_file_hash: bool = True,
) -> pd.DataFrame:
    """Verify that every expected node is represented by one valid cache."""

    params = parameters or PlaceMapCacheParameters()
    needs = collect_context_neuron_needs(pool, strict_ed6=strict_ed6)
    directory = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    for context in sorted(needs):
        session_id = context.session_id
        ids = needs[context]
        cache_path, manifest_path = context_cache_paths(directory, context, ids, params)
        if not cache_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"missing cache or manifest for {context.context_key}")
        arrays, manifest = load_session_map_cache(
            cache_path,
            manifest_path,
            expected_session_id=session_id,
            expected_context=context,
            expected_neuron_ids=ids,
            expected_parameters=params,
            verify_file_hash=bool(verify_file_hash),
        )
        rows.append(
            {
                "session_id": session_id,
                "context_key": context.context_key,
                "neural_variant": context.neural_variant,
                "n_neurons": int(len(arrays.neuron_ids)),
                "cache_file": cache_path.name,
                "manifest_file": manifest_path.name,
                "parameter_hash": manifest["parameter_hash"],
                "neuron_id_hash": manifest["neuron_id_hash"],
                "cache_file_sha256": manifest["cache_file_sha256"],
            }
        )
    index = pd.DataFrame(rows).sort_values("context_key", kind="mergesort").reset_index(drop=True)
    if int(index["n_neurons"].sum()) != int(sum(map(len, needs.values()))):
        raise AssertionError("assembled cache index does not cover every expected node")
    index.to_csv(directory / "cache_index_complete.csv", index=False)
    return index


def verify_vectorized_against_scalar(
    c_matrix: np.ndarray,
    selected_df: pd.DataFrame,
    neuron_id: int,
    scalar_compute_map: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    parameters: PlaceMapCacheParameters | None = None,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Numerically compare one vectorized map with ``compute_map_from_vectors``."""

    params = parameters or PlaceMapCacheParameters()
    vectorized = compute_session_maps_vectorized(
        c_matrix,
        selected_df,
        [int(neuron_id)],
        parameters=params,
        chunk_size=1,
        output_dtype=np.float64,
    )
    frames, positions, _ = _selected_frames_and_positions(
        selected_df.reset_index(drop=True), c_matrix, params
    )
    scalar_map, scalar_occupancy = scalar_compute_map(
        np.asarray(c_matrix[int(neuron_id), frames], dtype=float),
        positions,
        n_bins=int(params.n_bins),
        sigma=float(params.sigma),
        min_time_base=float(params.min_time_base),
        smooth=True,
    )
    vector_map = np.asarray(vectorized.smooth_maps[0], dtype=float)
    vector_mask = np.asarray(vectorized.occupancy_masks[0], dtype=bool)
    scalar_mask = np.asarray(scalar_occupancy, dtype=float) > 0
    max_abs_difference = float(np.max(np.abs(vector_map - np.asarray(scalar_map, dtype=float))))
    mask_equal = bool(np.array_equal(vector_mask, scalar_mask))
    map_equal = bool(np.allclose(vector_map, scalar_map, rtol=0.0, atol=float(atol)))
    if not mask_equal or not map_equal:
        raise AssertionError(
            "vectorized/scalar map mismatch: "
            f"mask_equal={mask_equal}, max_abs_difference={max_abs_difference:.3g}, atol={atol:.3g}"
        )
    return {
        "neuron_id": int(neuron_id),
        "mask_equal": mask_equal,
        "map_allclose": map_equal,
        "max_abs_difference": max_abs_difference,
        "absolute_tolerance": float(atol),
        "n_occupied_bins": int(vector_mask.sum()),
        "parameter_hash": parameter_hash(params),
    }


def verify_file_node_against_scalar(
    session_id: str,
    neuron_id: int,
    scalar_compute_map: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    behav_wd: Path | str,
    neural_wd: Path | str,
    parameters: PlaceMapCacheParameters | None = None,
    max_gap_frames: int = 64,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Run scalar parity while reading just one real session-neuron C trace."""

    params = parameters or PlaceMapCacheParameters()
    paths = resolve_session_input_paths(session_id, behav_wd=behav_wd, neural_wd=neural_wd)
    selected_df = pd.read_csv(
        paths.selected_csv,
        usecols=[params.frame_column, *params.position_columns],
    )
    frames = pd.to_numeric(selected_df[params.frame_column], errors="raise").to_numpy(dtype=int)
    positions = selected_df.loc[:, list(params.position_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    with h5py.File(paths.neural_mat, "r") as handle:
        if "C" not in handle or not isinstance(handle["C"], h5py.Dataset):
            raise RuntimeError(f"session {session_id} MAT has no dense C dataset")
        dataset = handle["C"]
        counts = read_selected_c_block(
            dataset,
            [int(neuron_id)],
            frames,
            max_gap_frames=int(max_gap_frames),
        )[0]
        vectorized = _compute_maps_with_count_loader(
            selected_df,
            [int(neuron_id)],
            c_shape=(int(dataset.shape[1]), int(dataset.shape[0])),
            count_loader=lambda _block_ids, requested_frames: read_selected_c_block(
                dataset,
                [int(neuron_id)],
                requested_frames,
                max_gap_frames=int(max_gap_frames),
            ),
            parameters=params,
            chunk_size=1,
            output_dtype=np.float64,
        )
        stored_c_shape = [int(dataset.shape[0]), int(dataset.shape[1])]
    scalar_map, scalar_occupancy = scalar_compute_map(
        counts,
        positions,
        n_bins=int(params.n_bins),
        sigma=float(params.sigma),
        min_time_base=float(params.min_time_base),
        smooth=True,
    )
    vector_map = np.asarray(vectorized.smooth_maps[0], dtype=float)
    vector_mask = np.asarray(vectorized.occupancy_masks[0], dtype=bool)
    scalar_mask = np.asarray(scalar_occupancy, dtype=float) > 0
    max_abs_difference = float(np.max(np.abs(vector_map - np.asarray(scalar_map, dtype=float))))
    mask_equal = bool(np.array_equal(vector_mask, scalar_mask))
    map_equal = bool(np.allclose(vector_map, scalar_map, rtol=0.0, atol=float(atol)))
    if not mask_equal or not map_equal:
        raise AssertionError(
            "selective-HDF5 vectorized/scalar map mismatch: "
            f"mask_equal={mask_equal}, max_abs_difference={max_abs_difference:.3g}, atol={atol:.3g}"
        )
    return {
        "session_id": str(session_id),
        "neuron_id": int(neuron_id),
        "signal_source": "C",
        "stored_c_shape_frames_by_neurons": stored_c_shape,
        "hdf5_datasets_read": ["C"],
        "selected_csv_columns_read": [params.frame_column, *params.position_columns],
        "mask_equal": mask_equal,
        "map_allclose": map_equal,
        "max_abs_difference": max_abs_difference,
        "absolute_tolerance": float(atol),
        "n_occupied_bins": int(vector_mask.sum()),
        "parameter_hash": parameter_hash(params),
    }


__all__ = [
    "ALGORITHM_VERSION",
    "CACHE_SCHEMA_VERSION",
    "EXPECTED_CANDIDATE_ROWS",
    "EXPECTED_CONTEXT_COUNT",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_SESSION_COUNT",
    "EXPECTED_UNIQUE_NODES",
    "NEURAL_VARIANT_ORDINARY",
    "NEURAL_VARIANT_PRESORT",
    "NEURAL_VARIANT_SORTED",
    "NeuralSessionContext",
    "PlaceMapCacheParameters",
    "SessionMapArrays",
    "SessionInputPaths",
    "assemble_exhaustive_cache_index",
    "build_exhaustive_session_cache",
    "build_context_neuron_index",
    "build_session_neuron_index",
    "collect_context_neuron_needs",
    "collect_session_neuron_needs",
    "compute_context_maps_from_files",
    "compute_session_maps_vectorized",
    "compute_session_maps_from_files",
    "context_cache_paths",
    "load_context_map_cache",
    "load_session_map_cache",
    "neuron_id_hash",
    "parameter_hash",
    "parameter_payload",
    "read_selected_c_block",
    "resolve_candidate_side_context",
    "resolve_context_input_paths",
    "resolve_session_input_paths",
    "session_cache_paths",
    "verify_vectorized_against_scalar",
    "verify_file_node_against_scalar",
    "write_session_map_cache",
]
