"""Maze level geometry used to draw representation schematics."""
from __future__ import annotations

import numpy as np

from .maze import NewMaze

N_BINS_X = 15
LEVEL_DEFINITION_GEOMETRIC8 = "geometric8"
LEVEL_DEFINITION_TREE_DEPTH7 = "tree_depth7"


def generate_level_list(ma_ce, grid_size=N_BINS_X):
    """
    Generate eight hierarchical level masks over the maze occupancy map.
    """
    n_cells = len(ma_ce)
    levels = [np.zeros(n_cells, dtype=int) for _ in range(8)]

    for i in range(0, 7):
        levels[0][ma_ce[(i, 7)]] = 1

    for j in range(4, 11):
        levels[1][ma_ce[(7, j)]] = 1

    for i in range(4, 11):
        for j in [3, 11]:
            levels[2][ma_ce[(i, j)]] = 1

    for i in [3, 11]:
        for j in [3, 11]:
            for dj in [0, -1, 1]:
                idx = ma_ce.get((i, j + dj))
                if idx is not None:
                    levels[3][idx] = 1

    for i in [3, 11]:
        for j in range(1, 14, 4):
            for di in [0, -1, 1]:
                idx = ma_ce.get((i + di, j))
                if idx is not None:
                    levels[4][idx] = 1

    for i in range(1, grid_size, 4):
        for j in range(1, grid_size, 4):
            levels[5][ma_ce[(i, j)]] = 1

    for i in range(1, grid_size, 4):
        for j in range(0, grid_size, 2):
            levels[6][ma_ce[(i, j)]] = 1

    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            levels[7][ma_ce[(i, j)]] = 1

    return levels

def normalize_level_definition(level_definition: str | None = None) -> str:
    """Normalize accepted names for maze level definitions."""
    text = str(level_definition or LEVEL_DEFINITION_GEOMETRIC8).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "geometric": LEVEL_DEFINITION_GEOMETRIC8,
        "geometric8": LEVEL_DEFINITION_GEOMETRIC8,
        "old_geometric": LEVEL_DEFINITION_GEOMETRIC8,
        "old_geometric8": LEVEL_DEFINITION_GEOMETRIC8,
        "8": LEVEL_DEFINITION_GEOMETRIC8,
        "8_level": LEVEL_DEFINITION_GEOMETRIC8,
        "level8": LEVEL_DEFINITION_GEOMETRIC8,
        "tree_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "tree_depth7": LEVEL_DEFINITION_TREE_DEPTH7,
        "true_tree_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "true_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "7": LEVEL_DEFINITION_TREE_DEPTH7,
        "7_level": LEVEL_DEFINITION_TREE_DEPTH7,
        "level7": LEVEL_DEFINITION_TREE_DEPTH7,
    }
    if text not in aliases:
        raise ValueError(
            "level_definition must be one of "
            f"{LEVEL_DEFINITION_GEOMETRIC8!r} or {LEVEL_DEFINITION_TREE_DEPTH7!r}; "
            f"got {level_definition!r}."
        )
    return aliases[text]

def generate_tree_depth_level_list(maze=None):
    """
    Generate seven true tree-depth masks over the maze occupancy map.

    The depth is the binary-tree run depth for each maze cell's owning run.
    For ``NewMaze(6)`` this covers all 176 maze cells with counts
    ``8, 8, 16, 16, 32, 32, 64`` for depths ``0..6``.
    """
    ma = NewMaze(6) if maze is None else maze
    run_parent = np.asarray(ma.pa, dtype=int).ravel()
    run_depth = np.full(run_parent.shape[0], -1, dtype=int)
    for run_idx in range(run_parent.shape[0]):
        depth = 0
        cur = int(run_idx)
        guard = 0
        while cur >= 0 and guard <= run_parent.shape[0] + 2:
            cur = int(run_parent[cur])
            if cur >= 0:
                depth += 1
            guard += 1
        run_depth[run_idx] = int(depth)

    n_levels = int(np.nanmax(run_depth)) + 1
    levels = [np.zeros(len(ma.xc), dtype=int) for _ in range(n_levels)]
    for cell_id, run_idx in enumerate(np.asarray(ma.rc, dtype=int).ravel()):
        if 0 <= int(run_idx) < run_depth.shape[0]:
            levels[int(run_depth[int(run_idx)])][int(cell_id)] = 1
    return levels

def get_level_list(level_definition: str | None = None, *, maze=None, ma_ce=None, grid_size=N_BINS_X):
    """Return level masks for the requested maze level definition."""
    definition = normalize_level_definition(level_definition)
    if definition == LEVEL_DEFINITION_GEOMETRIC8:
        if ma_ce is None:
            ma_ce = (NewMaze(6) if maze is None else maze).ce
        return generate_level_list(ma_ce, grid_size=grid_size)
    return generate_tree_depth_level_list(NewMaze(6) if maze is None else maze)
