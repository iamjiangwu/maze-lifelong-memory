"""Maze level geometry used to draw representation schematics."""
from __future__ import annotations

import numpy as np

from .maze import NewMaze


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


def get_level_list(level_definition: str | None = None, *, maze=None):
    """Return the seven tree-depth masks used by the included schematic."""
    if level_definition != "tree_depth7":
        raise ValueError("The included level schematic uses tree_depth7.")
    return generate_tree_depth_level_list(NewMaze(6) if maze is None else maze)
