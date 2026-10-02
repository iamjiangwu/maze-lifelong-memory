"""Array collection and ordering helpers for level-code heatmap assembly."""

from __future__ import annotations

from typing import Any

import numpy as np


def stack_level_maps(
    maps_by_pair: dict[str, Any],
    *,
    sort_keys: bool = False,
    return_slices: bool = False,
    expected_n_levels: int | None = 8,
) -> tuple[np.ndarray, dict[str, tuple[int, int]] | None]:
    """Stack ``(level, neuron)`` maps and optionally retain pair slices."""

    keys = sorted(maps_by_pair) if sort_keys else list(maps_by_pair)
    blocks: list[np.ndarray] = []
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    n_levels = None if expected_n_levels is None else int(expected_n_levels)
    for key in keys:
        block = np.asarray(maps_by_pair[key])
        if block.ndim != 2:
            raise ValueError(
                f"{key}: expected 2D array, got {block.ndim}D with shape {block.shape}"
            )
        if n_levels is None:
            n_levels = int(block.shape[0])
        if block.shape[0] != n_levels:
            raise ValueError(
                f"{key}: first dimension must be {n_levels}, got {block.shape[0]}"
            )
        blocks.append(block)
        end = start + int(block.shape[1])
        if return_slices:
            slices[key] = (start, end)
        start = end
    if not blocks:
        raise ValueError("No level maps were available to stack")
    stacked = np.concatenate(blocks, axis=1)
    return stacked, slices if return_slices else None


def normalize_maps(maps: Any, mode: str = "l2", eps: float = 1e-12) -> np.ndarray:
    """Normalize each neuron column using the maintained legacy definitions."""

    values = np.asarray(maps, dtype=float)
    if values.ndim != 2:
        raise ValueError("maps must be 2D (levels, neurons)")
    if mode == "l2":
        scale = np.linalg.norm(values, axis=0)
        return values / np.maximum(scale, eps)
    if mode == "max":
        return values / np.maximum(np.max(values, axis=0), eps)
    if mode == "sum1":
        return values / np.maximum(np.sum(values, axis=0), eps)
    if mode == "z":
        mean = np.mean(values, axis=0)
        std = np.maximum(np.std(values, axis=0), eps)
        return (values - mean) / std
    raise ValueError(f"Unknown mode: {mode}")
