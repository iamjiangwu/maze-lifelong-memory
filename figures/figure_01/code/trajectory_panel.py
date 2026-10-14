"""Exact frozen ED1 trajectory and icon definitions; see extraction provenance."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from ca_analysis.maze import NewMaze


GRAY_BLOCKS = (
    (-0.5, 0.5, 1, 1), (-0.5, 4.5, 1, 1), (-0.5, 8.5, 1, 1), (-0.5, 12.5, 1, 1),
    (2.5, 13.5, 1, 1), (6.5, 13.5, 1, 1), (10.5, 13.5, 1, 1),
    (13.5, 12.5, 1, 1), (13.5, 8.5, 1, 1), (13.5, 4.5, 1, 1), (13.5, 0.5, 1, 1),
    (10.5, -0.5, 1, 1), (6.5, -0.5, 1, 1), (2.5, -0.5, 1, 1),
    (6.5, 1.5, 1, 1), (6.5, 11.5, 1, 1), (10.5, 5.5, 1, 1), (10.5, 7.5, 1, 1),
    (5.5, 4.5, 1, 1), (5.5, 8.5, 1, 1), (7.5, 4.5, 1, 1), (7.5, 8.5, 1, 1),
    (2.5, 5.5, 1, 1), (2.5, 7.5, 1, 1),
    (-0.5, 2.5, 3, 1), (-0.5, 10.5, 3, 1), (11.5, 10.5, 3, 1), (11.5, 2.5, 3, 1),
    (5.5, 0.5, 3, 1), (5.5, 12.5, 3, 1), (7.5, 6.5, 7, 1),
)


def _draw_house_icon(ax) -> None:
    x, y, size = -1.33, 6.86, 0.88
    ax.add_patch(Rectangle((x + 0.23 * size, y - 0.48 * size), 0.14 * size, 0.28 * size,
                           facecolor="#111111", edgecolor="#111111", linewidth=0, zorder=8, clip_on=False))
    ax.add_patch(Polygon([
        (x - 0.56 * size, y - 0.03 * size), (x, y - 0.62 * size),
        (x + 0.56 * size, y - 0.03 * size), (x + 0.40 * size, y - 0.03 * size),
        (x + 0.40 * size, y + 0.48 * size), (x - 0.40 * size, y + 0.48 * size),
        (x - 0.40 * size, y - 0.03 * size),
    ], closed=True, facecolor="#111111", edgecolor="#111111", linewidth=0,
        zorder=8, clip_on=False, joinstyle="miter"))
    ax.add_patch(Rectangle((x - 0.10 * size, y + 0.19 * size), 0.20 * size, 0.29 * size,
                           facecolor="white", edgecolor="white", linewidth=0, zorder=8.1, clip_on=False))


def _draw_water_drop_icon(ax) -> None:
    x, y, size = 15.18, 6.12, 0.82
    verts = [
        (x, y - 0.66 * size), (x - 0.60 * size, y - 0.05 * size),
        (x - 0.48 * size, y + 0.52 * size), (x, y + 0.58 * size),
        (x + 0.48 * size, y + 0.52 * size), (x + 0.60 * size, y - 0.05 * size),
        (x, y - 0.66 * size), (x, y - 0.66 * size),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="#36a3ff", edgecolor="#0866b2",
                           linewidth=0.9, zorder=8, clip_on=False, joinstyle="round"))
    ax.add_patch(Ellipse((x - 0.13 * size, y - 0.02 * size), 0.15 * size, 0.30 * size,
                         angle=-28, facecolor="#c9ecff", edgecolor="none", alpha=0.9,
                         zorder=8.1, clip_on=False))


# Preserve viridis hues, gently darkening its bright end for thin lines on white.
_color_position = np.linspace(0.0, 1.0, 256)
_trajectory_colors = plt.get_cmap("viridis")(_color_position)
_trajectory_colors[:, :3] *= (1.0 - 0.22 * _color_position**2)[:, None]
TRAJECTORY_CMAP = LinearSegmentedColormap.from_list(
    "viridis_muted_yellow", _trajectory_colors
)


def draw_trajectory(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    norm: Normalize,
) -> None:
    maze = NewMaze(6)
    ax.plot(maze.wa[:, 0], maze.wa[:, 1], color="black", linewidth=0.62, zorder=2)
    for x, y, width, height in GRAY_BLOCKS:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.35,
                edgecolor="#D8D8D8",
                facecolor="#D8D8D8",
                zorder=1,
            )
        )
    points = data[["x", "y"]].to_numpy(float)
    elapsed = data["elapsed_minutes"].to_numpy(float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    keep = data["draw_to_next"].iloc[:-1].astype(bool).to_numpy()
    collection = LineCollection(
        segments[keep],
        cmap=TRAJECTORY_CMAP,
        norm=norm,
        linewidths=0.55,
        alpha=0.86,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(0.5 * (elapsed[:-1] + elapsed[1:])[keep])
    ax.add_collection(collection)
    _draw_house_icon(ax)
    _draw_water_drop_icon(ax)
    ax.set_xlim(-1.85, 15.95)
    ax.set_ylim(15.0, -1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="normal", pad=1.0)
