"""Representation-aware descriptors for ED6 example place maps.

This module is deliberately free of file-system and plotting operations.  It
turns the independently q99-normalized 45 x 45 display maps used by the ED6
gallery into auditable *display-field* descriptors.  These thresholded display
fields are visual-selection aids; they are not shuffle-significant place fields.

Coordinate contract
-------------------
``map[row, col]`` uses image coordinates with ``row`` increasing downward.
For the 45-bin display and 15-cell maze, ``maze_x = col // 3`` and
``maze_y = row // 3``.  Maze-wall coordinates are cell-centred, so the exact
affine transform into imshow pixel-centre coordinates is
``fine = (wall + 0.5) * 3 - 0.5`` (equivalently ``3 * wall + 1``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import label as ndi_label

from .maze import Maze, NewMaze


DISPLAY_N_BINS = 45
MAZE_GRID_SIZE = 15
DISPLAY_FIELD_THRESHOLD = 0.40
DISPLAY_FIELD_MIN_FINE_BINS = 3


@dataclass(frozen=True)
class FineMazeCrosswalk:
    """Per-fine-bin metadata for a ``NewMaze(6)`` display grid."""

    n_bins: int
    maze_grid_size: int
    bins_per_cell: int
    maze_x: np.ndarray
    maze_y: np.ndarray
    cell_id: np.ndarray
    run_index: np.ndarray
    tree_depth: np.ndarray
    display_level: np.ndarray
    branch_ordinal_0based: np.ndarray
    branch_ordinal_1based: np.ndarray
    branch_path: np.ndarray
    coarse_within_run_index: np.ndarray
    within_run_index: np.ndarray
    fine_within_run_index: np.ndarray
    within_run_fraction: np.ndarray
    within_run_region: np.ndarray
    maze_support: np.ndarray

    def record_at(self, row: int, col: int) -> dict[str, Any]:
        """Return scalar crosswalk metadata for one fine-bin index."""

        row = int(row)
        col = int(col)
        if not (0 <= row < self.n_bins and 0 <= col < self.n_bins):
            raise IndexError(f"fine bin ({row}, {col}) is outside 0..{self.n_bins - 1}")
        return {
            "row": row,
            "col": col,
            "maze_x": int(self.maze_x[row, col]),
            "maze_y": int(self.maze_y[row, col]),
            "cell_id": int(self.cell_id[row, col]),
            "run_index": int(self.run_index[row, col]),
            "tree_depth": int(self.tree_depth[row, col]),
            "display_level": int(self.display_level[row, col]),
            "branch_ordinal_0based": int(self.branch_ordinal_0based[row, col]),
            "branch_ordinal_1based": int(self.branch_ordinal_1based[row, col]),
            "branch_path": str(self.branch_path[row, col]),
            "coarse_within_run_index": int(self.coarse_within_run_index[row, col]),
            "within_run_index": int(self.within_run_index[row, col]),
            "fine_within_run_index": int(self.fine_within_run_index[row, col]),
            "within_run_fraction": float(self.within_run_fraction[row, col]),
            "within_run_region": str(self.within_run_region[row, col]),
            "on_maze": bool(self.maze_support[row, col]),
        }


def _grid_scale(n_bins: int, maze_grid_size: int) -> int:
    n_bins = int(n_bins)
    maze_grid_size = int(maze_grid_size)
    if n_bins <= 0 or maze_grid_size <= 0 or n_bins % maze_grid_size:
        raise ValueError("n_bins must be a positive multiple of maze_grid_size")
    return n_bins // maze_grid_size


def maze_wall_to_fine_coordinates(
    wall_xy: np.ndarray,
    *,
    n_bins: int = DISPLAY_N_BINS,
    maze_grid_size: int = MAZE_GRID_SIZE,
) -> np.ndarray:
    """Map cell-centred maze-wall ``(x, y)`` coordinates to imshow coordinates."""

    values = np.asarray(wall_xy, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError("wall_xy must have final dimension 2")
    scale = _grid_scale(n_bins, maze_grid_size)
    return (values + 0.5) * float(scale) - 0.5


def fine_to_maze_wall_coordinates(
    fine_xy: np.ndarray,
    *,
    n_bins: int = DISPLAY_N_BINS,
    maze_grid_size: int = MAZE_GRID_SIZE,
) -> np.ndarray:
    """Inverse of :func:`maze_wall_to_fine_coordinates`."""

    values = np.asarray(fine_xy, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError("fine_xy must have final dimension 2")
    scale = _grid_scale(n_bins, maze_grid_size)
    return (values + 0.5) / float(scale) - 0.5


def _run_depths(maze: Maze) -> np.ndarray:
    parents = np.asarray(maze.pa, dtype=int).ravel()
    depths = np.full(parents.size, -1, dtype=int)
    for run_index in range(parents.size):
        depth = 0
        current = int(run_index)
        seen: set[int] = set()
        while int(parents[current]) >= 0:
            if current in seen:
                raise ValueError("maze run-parent graph contains a cycle")
            seen.add(current)
            current = int(parents[current])
            depth += 1
        depths[run_index] = depth
    return depths


def _run_paths(maze: Maze) -> tuple[str, ...]:
    parents = np.asarray(maze.pa, dtype=int).ravel()
    children = np.asarray(maze.ch, dtype=int)
    paths: list[str] = []
    for run_index in range(parents.size):
        current = int(run_index)
        bits: list[str] = []
        while int(parents[current]) >= 0:
            parent = int(parents[current])
            hits = np.flatnonzero(children[parent] == current)
            if hits.size != 1:
                raise ValueError("maze child table is inconsistent with run parents")
            bits.append(str(int(hits[0])))
            current = parent
        paths.append("root" if not bits else ".".join(reversed(bits)))
    return tuple(paths)


def _within_run_region(fraction: float, depth: int) -> str:
    if not np.isfinite(fraction):
        return "off_maze"
    if int(depth) == 0:
        return (
            "entrance"
            if fraction < 1.0 / 3.0
            else ("middle" if fraction < 2.0 / 3.0 else "junction")
        )
    return (
        "proximal"
        if fraction < 1.0 / 3.0
        else ("middle" if fraction < 2.0 / 3.0 else "distal")
    )


def build_fine_maze_crosswalk(
    *,
    n_bins: int = DISPLAY_N_BINS,
    maze_grid_size: int = MAZE_GRID_SIZE,
    maze: Maze | None = None,
) -> FineMazeCrosswalk:
    """Build the exact row/column -> physical cell/run/tree-depth crosswalk."""

    scale = _grid_scale(n_bins, maze_grid_size)
    ma = NewMaze(6) if maze is None else maze
    if int(np.max(ma.xc)) >= maze_grid_size or int(np.max(ma.yc)) >= maze_grid_size:
        raise ValueError("maze coordinates exceed maze_grid_size")

    rows, cols = np.indices((int(n_bins), int(n_bins)), dtype=int)
    maze_x = cols // scale
    maze_y = rows // scale
    shape = rows.shape
    cell_id = np.full(shape, -1, dtype=int)
    for (x_cell, y_cell), value in ma.ce.items():
        cell_id[(maze_x == int(x_cell)) & (maze_y == int(y_cell))] = int(value)
    support = cell_id >= 0

    run_depth = _run_depths(ma)
    run_paths = _run_paths(ma)
    ordinals = np.full(len(ma.ru), -1, dtype=int)
    for depth in np.unique(run_depth):
        members = np.flatnonzero(run_depth == int(depth))
        ordinals[members] = np.arange(members.size, dtype=int)

    run_index = np.full(shape, -1, dtype=int)
    tree_depth = np.full(shape, -1, dtype=int)
    display_level = np.full(shape, -1, dtype=int)
    branch_0 = np.full(shape, -1, dtype=int)
    branch_1 = np.full(shape, -1, dtype=int)
    branch_path = np.full(shape, "off_maze", dtype=object)
    within_index = np.full(shape, -1, dtype=int)
    fine_within_index = np.full(shape, -1, dtype=int)
    within_fraction = np.full(shape, np.nan, dtype=float)
    within_region = np.full(shape, "off_maze", dtype=object)

    for run_id, cells in enumerate(ma.ru):
        depth = int(run_depth[run_id])
        if not cells:
            raise ValueError(f"maze run {run_id} has no owned cells")
        if run_id == 0:
            if len(cells) < 2:
                raise ValueError(
                    "root run needs at least two cells to define direction"
                )
            origin_cell = int(cells[0])
            next_cell = int(cells[1])
        else:
            parent_run = int(np.asarray(ma.pa, dtype=int)[run_id])
            if parent_run < 0:
                raise ValueError(f"non-root run {run_id} has no parent")
            origin_cell = int(ma.ru[parent_run][-1])
            next_cell = int(cells[0])
        direction_x = int(ma.xc[next_cell]) - int(ma.xc[origin_cell])
        direction_y = int(ma.yc[next_cell]) - int(ma.yc[origin_cell])
        if abs(direction_x) + abs(direction_y) != 1:
            raise ValueError(
                f"run {run_id} direction must be one cardinal-cell step; "
                f"got ({direction_x}, {direction_y})"
            )
        for position, cell in enumerate(cells):
            hit = cell_id == int(cell)
            local_x = cols[hit] % scale
            local_y = rows[hit] % scale
            if direction_x > 0:
                sub_position = local_x
            elif direction_x < 0:
                sub_position = (scale - 1) - local_x
            elif direction_y > 0:
                sub_position = local_y
            else:
                sub_position = (scale - 1) - local_y
            fine_position = int(scale) * int(position) + sub_position
            fraction = (fine_position.astype(float) + 0.5) / float(
                int(scale) * len(cells)
            )
            run_index[hit] = int(run_id)
            tree_depth[hit] = depth
            display_level[hit] = depth + 1
            branch_0[hit] = int(ordinals[run_id])
            branch_1[hit] = int(ordinals[run_id]) + 1
            branch_path[hit] = run_paths[run_id]
            within_index[hit] = int(position)
            fine_within_index[hit] = fine_position
            within_fraction[hit] = fraction
            within_region[hit] = np.asarray(
                [_within_run_region(value, depth) for value in fraction],
                dtype=object,
            )

    if np.any(support & (run_index < 0)):
        raise AssertionError("at least one physical maze cell lacks run metadata")
    return FineMazeCrosswalk(
        n_bins=int(n_bins),
        maze_grid_size=int(maze_grid_size),
        bins_per_cell=int(scale),
        maze_x=maze_x,
        maze_y=maze_y,
        cell_id=cell_id,
        run_index=run_index,
        tree_depth=tree_depth,
        display_level=display_level,
        branch_ordinal_0based=branch_0,
        branch_ordinal_1based=branch_1,
        branch_path=branch_path,
        coarse_within_run_index=within_index,
        within_run_index=within_index,
        fine_within_run_index=fine_within_index,
        within_run_fraction=within_fraction,
        within_run_region=within_region,
        maze_support=support,
    )


def _mass_shares(labels: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    finite = (labels >= 0) & np.isfinite(weights)
    if not np.any(finite):
        return {}
    values = np.clip(np.asarray(weights[finite], dtype=float), 0.0, None)
    if float(values.sum()) <= 0.0:
        values = np.ones(values.size, dtype=float)
    unique = np.unique(labels[finite])
    totals = {int(key): float(values[labels[finite] == key].sum()) for key in unique}
    denominator = float(sum(totals.values()))
    return {key: value / denominator for key, value in sorted(totals.items())}


def _share_token(shares: Mapping[int, float], prefix: str) -> str:
    if not shares:
        return "none"
    return "|".join(f"{prefix}{key}={value:.6f}" for key, value in shares.items())


def _count_category(count: int) -> str:
    count = int(count)
    return str(count) if count <= 3 else "4+"


def _phenotype(field_count: int, encoded_levels: tuple[int, ...]) -> tuple[str, str]:
    if field_count <= 0:
        return "no_field", "no_field"
    if field_count == 1:
        return "single_field", "single_field"
    same = len(encoded_levels) == 1
    stem = "same_level" if same else "multi_level"
    suffix = "2" if field_count == 2 else "3plus"
    return stem, f"{stem}_{suffix}"


def _validate_display_inputs(
    normalized_map: np.ndarray,
    raw_smoothed_c: np.ndarray,
    common_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.asarray(normalized_map, dtype=float)
    raw = np.asarray(raw_smoothed_c, dtype=float)
    common = np.asarray(common_mask, dtype=bool)
    if (
        normalized.ndim != 2
        or normalized.shape != raw.shape
        or raw.shape != common.shape
    ):
        raise ValueError(
            "normalized_map, raw_smoothed_c, and common_mask must be same-shape 2D arrays"
        )
    valid = common & np.isfinite(normalized)
    if np.any(valid & ~np.isfinite(raw)):
        raise ValueError(
            "raw_smoothed_c must be finite wherever common normalized_map is finite"
        )
    if np.any(normalized[valid] < -1e-9) or np.any(normalized[valid] > 1.0 + 1e-9):
        raise ValueError("normalized_map must be clipped to [0, 1] on common support")
    return normalized, raw, valid


def _field_record(
    field_id: int,
    rows: np.ndarray,
    cols: np.ndarray,
    normalized: np.ndarray,
    raw: np.ndarray,
    crosswalk: FineMazeCrosswalk,
) -> dict[str, Any]:
    raw_values = raw[rows, cols]
    norm_values = normalized[rows, cols]
    integrated_raw = float(np.sum(raw_values))
    peak_raw = float(np.max(raw_values))
    peak_hits = np.flatnonzero(raw_values == peak_raw)
    peak_order = np.lexsort((cols[peak_hits], rows[peak_hits]))
    peak_index = int(peak_hits[int(peak_order[0])])
    peak_row = int(rows[peak_index])
    peak_col = int(cols[peak_index])

    weights = np.clip(raw_values, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.clip(norm_values, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(rows.size, dtype=float)

    levels = crosswalk.display_level[rows, cols]
    runs = crosswalk.run_index[rows, cols]
    level_shares = _mass_shares(levels, weights)
    run_shares = _mass_shares(runs, weights)
    dominant_level = min(level_shares, key=lambda key: (-level_shares[key], int(key)))
    dominant_level_hit = levels == dominant_level
    within_level_run_shares = _mass_shares(
        runs[dominant_level_hit], weights[dominant_level_hit]
    )
    dominant_run = min(
        within_level_run_shares,
        key=lambda key: (-within_level_run_shares[key], int(key)),
    )
    dominant_hit = dominant_level_hit & (runs == dominant_run)
    dominant_fraction = float(
        np.average(
            crosswalk.within_run_fraction[rows[dominant_hit], cols[dominant_hit]],
            weights=weights[dominant_hit],
        )
    )
    dominant_first = int(np.flatnonzero(dominant_hit)[0])
    dominant_row = int(rows[dominant_first])
    dominant_col = int(cols[dominant_first])
    dominant_depth = int(crosswalk.tree_depth[dominant_row, dominant_col])
    if dominant_depth + 1 != int(dominant_level):
        raise AssertionError("dominant run is outside dominant display level")
    dominant_branch_0 = int(crosswalk.branch_ordinal_0based[dominant_row, dominant_col])
    dominant_branch_1 = dominant_branch_0 + 1
    dominant_path = str(crosswalk.branch_path[dominant_row, dominant_col])
    dominant_region = _within_run_region(dominant_fraction, dominant_depth)
    location_token = f"L{dominant_level}:B{dominant_branch_1}:{dominant_region}"

    peak = crosswalk.record_at(peak_row, peak_col)
    return {
        "field_id": int(field_id),
        "component_area_fine_bins": int(rows.size),
        "integrated_raw_smoothed_c": integrated_raw,
        "peak_raw_smoothed_c": peak_raw,
        "peak_normalized": float(normalized[peak_row, peak_col]),
        "peak_row_45": peak_row,
        "peak_col_45": peak_col,
        "peak_maze_x_15": int(peak["maze_x"]),
        "peak_maze_y_15": int(peak["maze_y"]),
        "peak_cell_id": int(peak["cell_id"]),
        "peak_run_index": int(peak["run_index"]),
        "peak_tree_depth_0based": int(peak["tree_depth"]),
        "peak_display_level_1based": int(peak["display_level"]),
        "peak_branch_ordinal_0based": int(peak["branch_ordinal_0based"]),
        "peak_branch_ordinal_1based": int(peak["branch_ordinal_1based"]),
        "peak_branch_path": str(peak["branch_path"]),
        "peak_coarse_within_run_index_0based": int(peak["coarse_within_run_index"]),
        "peak_within_run_index_0based": int(peak["within_run_index"]),
        "peak_fine_within_run_index_0based": int(peak["fine_within_run_index"]),
        "peak_within_run_fraction": float(peak["within_run_fraction"]),
        "peak_within_run_region": str(peak["within_run_region"]),
        "dominant_run_index": int(dominant_run),
        "dominant_tree_depth_0based": dominant_depth,
        "dominant_display_level_1based": int(dominant_level),
        "dominant_branch_ordinal_0based": dominant_branch_0,
        "dominant_branch_ordinal_1based": dominant_branch_1,
        "dominant_branch_path": dominant_path,
        "dominant_within_run_fraction": dominant_fraction,
        "dominant_within_run_region": dominant_region,
        "dominant_fine_within_run_fraction": dominant_fraction,
        "dominant_fine_within_run_region": dominant_region,
        "field_location_token": location_token,
        "level_mass_shares": level_shares,
        "level_mass_token": _share_token(level_shares, "L"),
        "run_mass_shares": run_shares,
        "run_mass_token": _share_token(run_shares, "R"),
    }


def _map_summary(fields: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(fields)
    encoded_levels = tuple(
        sorted({int(field["dominant_display_level_1based"]) for field in fields})
    )
    encoded_level_token = (
        "none"
        if not encoded_levels
        else "|".join(f"L{value}" for value in encoded_levels)
    )
    level_kind, phenotype = _phenotype(count, encoded_levels)
    locations = tuple(str(field["field_location_token"]) for field in fields)

    level_totals: dict[int, float] = {}
    run_totals: dict[int, float] = {}
    for field in fields:
        field_mass = max(float(field["integrated_raw_smoothed_c"]), 0.0)
        if field_mass <= 0.0:
            field_mass = float(field["component_area_fine_bins"])
        for key, share in field["level_mass_shares"].items():
            level_totals[int(key)] = level_totals.get(
                int(key), 0.0
            ) + field_mass * float(share)
        for key, share in field["run_mass_shares"].items():
            run_totals[int(key)] = run_totals.get(int(key), 0.0) + field_mass * float(
                share
            )
    level_mass = (
        _mass_shares(
            np.asarray(list(level_totals), dtype=int),
            np.asarray(list(level_totals.values()), dtype=float),
        )
        if level_totals
        else {}
    )
    run_mass = (
        _mass_shares(
            np.asarray(list(run_totals), dtype=int),
            np.asarray(list(run_totals.values()), dtype=float),
        )
        if run_totals
        else {}
    )
    signature = (
        f"{phenotype}|levels={encoded_level_token}|"
        f"locations={','.join(locations) if locations else 'none'}"
    )
    return {
        "field_count": int(count),
        "field_count_category": _count_category(count),
        "encoded_level_set": encoded_levels,
        "encoded_level_token": encoded_level_token,
        "same_or_multi_level": level_kind,
        "level_phenotype": phenotype,
        "field_location_tokens": locations,
        "field_location_token": "|".join(locations) if locations else "none",
        "level_mass_shares": level_mass,
        "level_mass_token": _share_token(level_mass, "L"),
        "run_mass_shares": run_mass,
        "run_mass_token": _share_token(run_mass, "R"),
        "representation_signature": signature,
        "has_single_field": count == 1,
        "has_two_fields": count == 2,
        "has_threeplus_fields": count >= 3,
        "is_same_level_multifield": count >= 2 and len(encoded_levels) == 1,
        "is_multi_level_multifield": count >= 2 and len(encoded_levels) > 1,
        **{f"encodes_level_{level}": level in encoded_levels for level in range(1, 8)},
    }


def describe_display_map(
    normalized_map: np.ndarray,
    raw_smoothed_c: np.ndarray,
    common_mask: np.ndarray,
    *,
    threshold: float = DISPLAY_FIELD_THRESHOLD,
    min_component_area: int = DISPLAY_FIELD_MIN_FINE_BINS,
    crosswalk: FineMazeCrosswalk | None = None,
) -> dict[str, Any]:
    """Extract and describe thresholded ED6 display fields.

    Components use 8-connectivity and are retained at three or more fine bins
    by default.  Retained components are deterministically ordered by decreasing
    integrated raw smoothed C, then decreasing raw peak, then peak row/column.
    The encoded-level set is the union of each field's dominant mass level;
    full cross-level spill is retained separately in ``level_mass_shares``.
    """

    normalized, raw, valid = _validate_display_inputs(
        normalized_map, raw_smoothed_c, common_mask
    )
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if int(min_component_area) <= 0:
        raise ValueError("min_component_area must be positive")
    cross = crosswalk or build_fine_maze_crosswalk(n_bins=normalized.shape[0])
    if normalized.shape != (cross.n_bins, cross.n_bins):
        raise ValueError("map shape does not match crosswalk")

    active = valid & (normalized >= float(threshold))
    if np.any(active & ~cross.maze_support):
        rows, cols = np.where(active & ~cross.maze_support)
        raise ValueError(
            "active common-support bins fall outside the physical maze; "
            f"first offending bin is ({int(rows[0])}, {int(cols[0])})"
        )
    original_labels, count = ndi_label(active, structure=np.ones((3, 3), dtype=int))
    components: list[dict[str, Any]] = []
    for original_id in range(1, int(count) + 1):
        rows, cols = np.where(original_labels == original_id)
        if rows.size < int(min_component_area):
            continue
        raw_values = raw[rows, cols]
        peak = float(np.max(raw_values))
        peak_hits = np.flatnonzero(raw_values == peak)
        order = np.lexsort((cols[peak_hits], rows[peak_hits]))
        peak_idx = int(peak_hits[int(order[0])])
        components.append(
            {
                "original_id": int(original_id),
                "rows": rows,
                "cols": cols,
                "integrated": float(np.sum(raw_values)),
                "peak": peak,
                "peak_row": int(rows[peak_idx]),
                "peak_col": int(cols[peak_idx]),
            }
        )
    components.sort(
        key=lambda item: (
            -float(item["integrated"]),
            -float(item["peak"]),
            int(item["peak_row"]),
            int(item["peak_col"]),
            int(item["original_id"]),
        )
    )

    labels = np.zeros(normalized.shape, dtype=np.int16)
    fields: list[dict[str, Any]] = []
    for field_id, component in enumerate(components, start=1):
        rows = np.asarray(component["rows"], dtype=int)
        cols = np.asarray(component["cols"], dtype=int)
        labels[rows, cols] = int(field_id)
        fields.append(_field_record(field_id, rows, cols, normalized, raw, cross))
    return {
        "field_labels": labels,
        "field_mask": labels > 0,
        "fields": fields,
        "summary": _map_summary(fields),
        "threshold": float(threshold),
        "min_component_area_fine_bins": int(min_component_area),
        "connectivity": 8,
        "field_definition": "q99_normalized_common_support_display_field",
    }


def _set_relation(first: set[Any], second: set[Any]) -> str:
    if not first and not second:
        return "both_empty"
    if not first:
        return "gain"
    if not second:
        return "loss"
    if first == second:
        return "same"
    if first.isdisjoint(second):
        return "disjoint"
    return "partial"


def summarize_display_pair(
    summary_a: Mapping[str, Any], summary_b: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize the semantic A -> B transition between two map summaries."""

    count_a = int(summary_a["field_count"])
    count_b = int(summary_b["field_count"])
    count_delta = count_b - count_a
    count_direction = (
        "gain" if count_delta > 0 else ("loss" if count_delta < 0 else "same")
    )
    levels_a = set(int(value) for value in summary_a["encoded_level_set"])
    levels_b = set(int(value) for value in summary_b["encoded_level_set"])
    locations_a = set(str(value) for value in summary_a["field_location_tokens"])
    locations_b = set(str(value) for value in summary_b["field_location_tokens"])
    level_relation = _set_relation(levels_a, levels_b)
    location_relation = _set_relation(locations_a, locations_b)
    field_transition = (
        f"{summary_a['field_count_category']}->{summary_b['field_count_category']}"
    )
    level_transition = (
        f"{summary_a['encoded_level_token']}->{summary_b['encoded_level_token']}"
    )
    location_transition = (
        f"{summary_a['field_location_token']}->{summary_b['field_location_token']}"
    )
    phenotype_transition = (
        f"{summary_a['level_phenotype']}->{summary_b['level_phenotype']}"
    )
    representation_transition = (
        f"{phenotype_transition}|fields={field_transition}:{count_direction}|"
        f"levels={level_relation}|locations={location_relation}"
    )
    return {
        "field_count_a": count_a,
        "field_count_b": count_b,
        "field_count_delta": int(count_delta),
        "field_count_direction": count_direction,
        "field_count_transition": field_transition,
        "level_set_transition": level_transition,
        "level_relation": level_relation,
        "location_set_transition": location_transition,
        "location_relation": location_relation,
        "phenotype_transition": phenotype_transition,
        "representation_transition": representation_transition,
    }


def describe_display_pair(
    normalized_a: np.ndarray,
    raw_a: np.ndarray,
    normalized_b: np.ndarray,
    raw_b: np.ndarray,
    common_mask: np.ndarray,
    *,
    threshold: float = DISPLAY_FIELD_THRESHOLD,
    min_component_area: int = DISPLAY_FIELD_MIN_FINE_BINS,
    crosswalk: FineMazeCrosswalk | None = None,
) -> dict[str, Any]:
    """Describe both sides and their semantic transition under one support mask."""

    cross = crosswalk or build_fine_maze_crosswalk(
        n_bins=np.asarray(normalized_a).shape[0]
    )
    side_a = describe_display_map(
        normalized_a,
        raw_a,
        common_mask,
        threshold=threshold,
        min_component_area=min_component_area,
        crosswalk=cross,
    )
    side_b = describe_display_map(
        normalized_b,
        raw_b,
        common_mask,
        threshold=threshold,
        min_component_area=min_component_area,
        crosswalk=cross,
    )
    return {
        "a": side_a,
        "b": side_b,
        "pair_summary": summarize_display_pair(side_a["summary"], side_b["summary"]),
    }


def field_count_neutral_map_quality(
    normalized_map: np.ndarray,
    common_mask: np.ndarray,
    *,
    active_threshold: float = DISPLAY_FIELD_THRESHOLD,
) -> dict[str, float]:
    """Score visible contrast/coherence without using field-count compactness.

    The activity term is a lower visibility gate that plateaus after nine fine
    bins.  There is no upper active-area bound, component-count term, or
    largest-component dominance term, so multiple separated coherent fields are
    not penalized merely for being multiple.
    """

    normalized = np.asarray(normalized_map, dtype=float)
    common = np.asarray(common_mask, dtype=bool)
    if normalized.ndim != 2 or normalized.shape != common.shape:
        raise ValueError("normalized_map and common_mask must be same-shape 2D arrays")
    if not 0.0 < float(active_threshold) <= 1.0:
        raise ValueError("active_threshold must be in (0, 1]")
    valid = common & np.isfinite(normalized)
    values = np.clip(normalized[valid], 0.0, 1.0)
    if values.size < 12 or float(np.max(values, initial=0.0)) <= 0.0:
        return {
            "contrast_score": 0.0,
            "active_region_coherence": 0.0,
            "visible_activity_score": 0.0,
            "active_fraction": 0.0,
            "active_pixel_count": 0.0,
            "image_quality_score": 0.0,
        }

    baseline = float(np.median(values))
    upper = float(np.quantile(values, 0.99))
    contrast = float(np.clip((upper - baseline) / 0.55, 0.0, 1.0))
    active = valid & (normalized >= float(active_threshold))
    active_count = int(active.sum())
    visibility = float(np.clip(active_count / 9.0, 0.0, 1.0))

    similarities: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        first = normalized[
            max(0, dy) : normalized.shape[0] + min(0, dy),
            max(0, dx) : normalized.shape[1] + min(0, dx),
        ]
        second = normalized[
            max(0, -dy) : normalized.shape[0] - max(0, dy),
            max(0, -dx) : normalized.shape[1] - max(0, dx),
        ]
        first_active = active[
            max(0, dy) : active.shape[0] + min(0, dy),
            max(0, dx) : active.shape[1] + min(0, dx),
        ]
        second_active = active[
            max(0, -dy) : active.shape[0] - max(0, dy),
            max(0, -dx) : active.shape[1] - max(0, dx),
        ]
        pair = first_active & second_active
        if np.any(pair):
            similarities.append(1.0 - np.abs(first[pair] - second[pair]))
    coherence = (
        float(np.clip(np.mean(np.concatenate(similarities)), 0.0, 1.0))
        if similarities
        else 0.0
    )
    base = 0.50 * contrast + 0.35 * coherence + 0.15 * visibility
    gate = np.sqrt(contrast * visibility)
    quality = float(np.clip(base * (0.50 + 0.50 * gate), 0.0, 1.0))
    return {
        "contrast_score": contrast,
        "active_region_coherence": coherence,
        "visible_activity_score": visibility,
        "active_fraction": float(active_count / max(int(valid.sum()), 1)),
        "active_pixel_count": float(active_count),
        "image_quality_score": quality,
    }


__all__ = [
    "DISPLAY_FIELD_MIN_FINE_BINS",
    "DISPLAY_FIELD_THRESHOLD",
    "DISPLAY_N_BINS",
    "MAZE_GRID_SIZE",
    "FineMazeCrosswalk",
    "build_fine_maze_crosswalk",
    "describe_display_map",
    "describe_display_pair",
    "field_count_neutral_map_quality",
    "fine_to_maze_wall_coordinates",
    "maze_wall_to_fine_coordinates",
    "summarize_display_pair",
]
