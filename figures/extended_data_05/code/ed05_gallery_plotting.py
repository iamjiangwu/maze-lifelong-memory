"""Draw the supplied ED5 map examples without raw-data reconstruction.

Maps are already normalized on the 45 x 45 display grid. Their labels come from
the retained 15 x 15 analysis correlations; drawing does not select examples or
recompute correlations.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from ca_analysis.maze import NewMaze


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


DISPLAY_N_BINS = 45
MAZE_GRID_SIZE = 15


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


def _format_place_map_correlation(value: float) -> str:
    return f"r = {float(value):.2f}".replace("-", "\N{MINUS SIGN}")
