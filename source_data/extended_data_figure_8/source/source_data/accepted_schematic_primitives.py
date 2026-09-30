"""Exact accepted schematic geometry/icon primitives; source attribution in INPUT_MANIFEST."""
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath

GRAY_BLOCKS = [
    [-0.5, 0.5, 1, 1],
    [-0.5, 4.5, 1, 1],
    [-0.5, 8.5, 1, 1],
    [-0.5, 12.5, 1, 1],
    [2.5, 13.5, 1, 1],
    [6.5, 13.5, 1, 1],
    [10.5, 13.5, 1, 1],
    [13.5, 12.5, 1, 1],
    [13.5, 8.5, 1, 1],
    [13.5, 4.5, 1, 1],
    [13.5, 0.5, 1, 1],
    [10.5, -0.5, 1, 1],
    [6.5, -0.5, 1, 1],
    [2.5, -0.5, 1, 1],
    [6.5, 1.5, 1, 1],
    [6.5, 11.5, 1, 1],
    [10.5, 5.5, 1, 1],
    [10.5, 7.5, 1, 1],
    [5.5, 4.5, 1, 1],
    [5.5, 8.5, 1, 1],
    [7.5, 4.5, 1, 1],
    [7.5, 8.5, 1, 1],
    [2.5, 5.5, 1, 1],
    [2.5, 7.5, 1, 1],
    [-0.5, 2.5, 3, 1],
    [-0.5, 10.5, 3, 1],
    [11.5, 10.5, 3, 1],
    [11.5, 2.5, 3, 1],
    [5.5, 0.5, 3, 1],
    [5.5, 12.5, 3, 1],
    [7.5, 6.5, 7, 1],
]

def _draw_house_icon(ax, spec: dict):
    x, y = map(float, spec["xy"])
    size = float(spec["size"])
    facecolor = spec.get("facecolor", "#111111")
    edgecolor = spec.get("edgecolor", facecolor)
    linewidth = float(spec.get("linewidth", 0.0))
    zorder = float(spec.get("zorder", 8))

    chimney = patches.Rectangle(
        (x + 0.23 * size, y - 0.48 * size),
        0.14 * size,
        0.28 * size,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
    )
    silhouette = patches.Polygon(
        [
            (x - 0.56 * size, y - 0.03 * size),
            (x, y - 0.62 * size),
            (x + 0.56 * size, y - 0.03 * size),
            (x + 0.40 * size, y - 0.03 * size),
            (x + 0.40 * size, y + 0.48 * size),
            (x - 0.40 * size, y + 0.48 * size),
            (x - 0.40 * size, y - 0.03 * size),
        ],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
        joinstyle="miter",
    )
    door = patches.Rectangle(
        (x - 0.10 * size, y + 0.19 * size),
        0.20 * size,
        0.29 * size,
        facecolor=spec.get("door_color", "white"),
        edgecolor=spec.get("door_color", "white"),
        linewidth=0.0,
        zorder=zorder + 0.1,
        clip_on=False,
    )
    for patch in (chimney, silhouette, door):
        ax.add_patch(patch)

def _draw_water_drop_icon(ax, spec: dict):
    x, y = map(float, spec["xy"])
    size = float(spec["size"])
    facecolor = spec.get("facecolor", "#36a3ff")
    edgecolor = spec.get("edgecolor", "#0866b2")
    linewidth = float(spec.get("linewidth", 0.9))
    zorder = float(spec.get("zorder", 8))

    verts = [
        (x, y - 0.66 * size),
        (x - 0.60 * size, y - 0.05 * size),
        (x - 0.48 * size, y + 0.52 * size),
        (x, y + 0.58 * size),
        (x + 0.48 * size, y + 0.52 * size),
        (x + 0.60 * size, y - 0.05 * size),
        (x, y - 0.66 * size),
        (x, y - 0.66 * size),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    drop = patches.PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
        joinstyle="round",
    )
    highlight = patches.Ellipse(
        (x - 0.13 * size, y - 0.02 * size),
        width=0.15 * size,
        height=0.30 * size,
        angle=-28,
        facecolor=spec.get("highlight_color", "#c9ecff"),
        edgecolor="none",
        alpha=0.9,
        zorder=zorder + 0.1,
        clip_on=False,
    )
    ax.add_patch(drop)
    ax.add_patch(highlight)
