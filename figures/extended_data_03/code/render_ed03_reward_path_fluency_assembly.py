"""Assemble Extended Data Figure 3 (reward-path fluency and first-visit choices).

Layout authority is slide 4 of ``sup-figs-v2.pptx``. After Extended Data Figure 2
is merged into Extended Data Figure 1, that slide carries the Extended Data
Figure 3 role, so the panels are lettered a-f here.

Every panel is source-rendered from the maintained plot-ready tables; no panel is
copied as a raster crop. Deliberate corrections to the slide:

1. The slide's a-d region is a crop of ``ed04_fluency_four_panel_story_current``,
   which the panel manifest marks ``role=source`` ("navigation and visual-QA
   overview; individual panels are manuscript-facing"). The four claims are
   redrawn here through the maintained ``draw_claim_panel`` contract.
2. The slide's assay schematic is
   ``ed04_panel_b_reward_path_first_turn_schematic_source.png``, which lives in
   ``99_legacy_audit_only`` and encodes the superseded first-turn assay. Panel e
   instead draws the six scored decision points of the current first-visit route
   on the canonical main-figure maze schematic, suppressing only the coloured
   end-node quadrants and the long-bout path.
3. The slide's depth bars use a superseded warm palette. The maintained palette
   (``#8BBDEB``/``#5F8C62``/``#DA2323``/``#B3A99B``) is used instead, which is the
   same condition palette as Main Figure 1, and it is applied to every panel.
4. Typography follows the main-figure contract at printed size, background grid
   lines are off, and on-panel text is limited to what the caption cannot carry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis.figure_panels import (
    GRAY_BLOCKS,
    draw_gray_blocks,
    draw_maze_landmarks,
    draw_maze_outline,
    draw_quadrant_overlays,
    finish_panel,
    get_maze_schematic_landmarks,
    get_panel_maze,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFRESH_ROOT = REPO_ROOT / "docs" / "extended_data_figure_panel_refresh_2026-07-12"
SOURCE_TABLES = (
    REFRESH_ROOT
    / "extended_data_figures"
    / "extended_data_fig_04"
    / "04_source_tables"
)
ROUTE_TABLES = REFRESH_ROOT / "source_tables"
FLUENCY_ROWS = SOURCE_TABLES / "fluency_plot_rows.csv"
FLUENCY_STATS = SOURCE_TABLES / "fluency_statistics.csv"
DEPTH_TABLE = ROUTE_TABLES / "ed04_depth_small_multiples_current_plus_new12.csv"
ROUTE_DEFINITION = ROUTE_TABLES / "ed04_route_definition_current_plus_new12.csv"

# Deliberately not under source_rendered_new/extended_data_fig_03: that directory holds
# the OLD Extended Data Figure 3 (branch-policy / fixed-30-minute exploration), which is
# slide 3 and becomes Figure 2 under the merged numbering.
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed03_reward_path_fluency"
    / "assembly"
)

FORMATS = ("svg", "pdf", "png", "tiff")

FIGURE_WIDTH_IN = 7.5
FIGURE_HEIGHT_IN = 3.95

# Typography is matched to the main figures at printed size. Main Figure 1 is drawn on a
# 10-inch canvas and exported at 180 mm, so its drawn points shrink by 180/(10*25.4). This
# figure is drawn 7.5 inches wide for the same 180 mm export, so its drawn points shrink by
# 180/(7.5*25.4). Matching printed sizes therefore means drawing at 0.75x Figure 1's sizes.
EXPORT_WIDTH_MM = 180.0
MAIN_FIGURE_CANVAS_WIDTH_IN = 10.0
FONT_SCALE = FIGURE_WIDTH_IN / MAIN_FIGURE_CANVAS_WIDTH_IN  # = 0.75

FONT_FAMILY = "Arial"
MAIN_BASE_FONT_SIZE = 8.0
MAIN_AXIS_LABEL_SIZE = 8.0
MAIN_TICK_LABEL_SIZE = 7.0
MAIN_TITLE_SIZE = 8.0
MAIN_PANEL_ID_SIZE = 13.0
MAIN_AXES_LINEWIDTH = 0.8

BASE_FONT_SIZE = MAIN_BASE_FONT_SIZE * FONT_SCALE
AXIS_LABEL_SIZE = MAIN_AXIS_LABEL_SIZE * FONT_SCALE
TICK_LABEL_SIZE = MAIN_TICK_LABEL_SIZE * FONT_SCALE
TITLE_SIZE = MAIN_TITLE_SIZE * FONT_SCALE
PANEL_ID_SIZE = MAIN_PANEL_ID_SIZE * FONT_SCALE
AXES_LINEWIDTH = MAIN_AXES_LINEWIDTH * FONT_SCALE

# Condition palette, taken from Main Figure 1 and applied to every panel. The maintained
# ED4 fluency renderer ships its own slightly different return/latent reds, which made the
# same condition appear in two colours within one figure; those are overridden below.
FIGURE1_NAIVE_COLOR = "#8BBDEB"
FIGURE1_SECOND_DAY_COLOR = "#5F8C62"
FIGURE1_LATENT_DAY2_COLOR = "#B3A99B"

# Main Figure 1 uses #DA2323 for Return day. In this figure it reads too red: at saturation
# 0.84 / luminance 74 it is roughly twice as saturated and half as bright as its three
# palette-mates (0.13-0.41 / 127-182), and large filled bars amplify that far more than
# Figure 1's small scatter markers do. Softened to a muted red of comparable weight.
FIGURE1_RETURN_COLOR_MAIN = "#DA2323"
FIGURE1_RETURN_COLOR = "#E15759"

# Naive and Return are both day-1 conditions, so the "1" is dropped from their labels.
CONDITION_LABELS = {
    "naive": "Naive day",
    "experienced_day2": "Second day",
    "return": "Return day",
    "latent_day2": "Latent day2",
}
FLUENCY_CONDITION_ORDER = ("latent", "return")
DEPTH_PANEL_ORDER = ("naive", "experienced_day2", "return", "latent_day2")
DEPTH_PANEL_LABELS = CONDITION_LABELS
DEPTH_PANEL_COLORS = {
    "naive": FIGURE1_NAIVE_COLOR,
    "experienced_day2": FIGURE1_SECOND_DAY_COLOR,
    "return": FIGURE1_RETURN_COLOR,
    "latent_day2": FIGURE1_LATENT_DAY2_COLOR,
}
AXIS_COLOR = "#202020"
GRID_COLOR = "#D8D8D8"
LEGACY_COLOR = "#8A8A8A"
MAZE_WALL_COLOR = "#B4B4B4"
PATH_COLOR = "#202020"

# Reward path through the binary tree; the terminal run is the canonical reward node.
REWARD_PATH_RUNS = (0, 1, 4, 10, 22, 46, 94)

# Slide-derived geometry, in inches, measured from the figure's lower-left corner.
ROW1_PANEL_LEFTS = (0.42, 2.265, 4.11, 5.955)
ROW1_PANEL_WIDTH = 1.44
ROW1_BOTTOM = 2.15
ROW1_HEIGHT = 1.45

DEPTH_PANEL_LEFTS = (2.34, 3.61, 4.88, 6.15)
DEPTH_PANEL_WIDTH = 1.17
DEPTH_BOTTOM = 0.50
DEPTH_HEIGHT = 1.15

# Panel e is sized from the panel f bar-plot height and then scaled up; the maze aspect
# fixes its width, and it stays vertically centred on the panel f band.
SCHEMATIC_VIEW_X = (-2.2, 16.2)
SCHEMATIC_VIEW_Y = (-0.9, 15.4)
SCHEMATIC_SCALE = 1.25
_SCHEMATIC_ASPECT = (SCHEMATIC_VIEW_X[1] - SCHEMATIC_VIEW_X[0]) / (
    SCHEMATIC_VIEW_Y[1] - SCHEMATIC_VIEW_Y[0]
)
_SCHEMATIC_HEIGHT = DEPTH_HEIGHT * SCHEMATIC_SCALE
_SCHEMATIC_WIDTH = _SCHEMATIC_HEIGHT * _SCHEMATIC_ASPECT
# The column available to panel e: figure left margin to the panel f y-axis label. The
# schematic box is centred inside it, and the view window is centred on the maze, so the
# maze reads centred in the column rather than pushed toward panel f.
SCHEMATIC_COLUMN_IN = (0.10, 2.05)
SCHEMATIC_RECT_IN = (
    SCHEMATIC_COLUMN_IN[0]
    + (SCHEMATIC_COLUMN_IN[1] - SCHEMATIC_COLUMN_IN[0] - _SCHEMATIC_WIDTH) / 2.0,
    DEPTH_BOTTOM + (DEPTH_HEIGHT - _SCHEMATIC_HEIGHT) / 2.0,
    _SCHEMATIC_WIDTH,
    _SCHEMATIC_HEIGHT,
)

PANEL_LETTER_SIZE = PANEL_ID_SIZE


def configure_main_figure_typography() -> None:
    """Apply the main-figure typography contract, scaled to this canvas."""
    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "axes.linewidth": AXES_LINEWIDTH,
            "xtick.major.width": AXES_LINEWIDTH,
            "ytick.major.width": AXES_LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_fluency_module() -> Any:
    """Import the maintained ED4 fluency renderer for its drawing contract."""
    path = REPO_ROOT / "scripts" / "render_ed04_fluency_story_current.py"
    spec = importlib.util.spec_from_file_location("ed04_fluency_story", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rect(x: float, y: float, width: float, height: float) -> list[float]:
    """Convert an inch rectangle to a matplotlib figure-fraction rectangle."""
    return [
        x / FIGURE_WIDTH_IN,
        y / FIGURE_HEIGHT_IN,
        width / FIGURE_WIDTH_IN,
        height / FIGURE_HEIGHT_IN,
    ]


def panel_letter(fig: plt.Figure, x_in: float, y_in: float, letter: str) -> None:
    fig.text(
        x_in / FIGURE_WIDTH_IN,
        y_in / FIGURE_HEIGHT_IN,
        letter,
        ha="left",
        va="top",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
    )


def draw_claim_bar_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    claim: dict[str, str],
    fluency: Any,
    rng: np.random.Generator,
) -> None:
    """Draw one fluency claim in the Main Figure 1 bar-plus-scatter grammar.

    Bar heights and intervals are the maintained one-group estimates (mouse-clustered
    mean and 95% CI); only the visual grammar changes. Marker size, cap size and line
    weights are Figure 1's values scaled to this canvas.
    """
    estimates = fluency.one_group_estimates(rows).set_index("condition")
    colors = {
        "latent": FIGURE1_LATENT_DAY2_COLOR,
        "return": FIGURE1_RETURN_COLOR,
    }
    observed: list[float] = []
    for index, condition in enumerate(FLUENCY_CONDITION_ORDER):
        subset = rows.loc[rows["condition"].eq(condition)]
        values = subset["mean_decision_time_s"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        observed.extend(values.tolist())
        estimate = estimates.loc[condition]
        mean = float(estimate["mean"])
        low = float(estimate["ci95_low"])
        high = float(estimate["ci95_high"])
        observed.extend([low, high])

        ax.bar(
            index,
            mean,
            width=0.60,
            color=colors[condition],
            edgecolor="none",
            alpha=0.78,
            zorder=1,
        )
        ax.scatter(
            np.full(len(values), index, dtype=float) + rng.normal(0.0, 0.06, len(values)),
            values,
            s=15.0 * FONT_SCALE**2,
            facecolor=colors[condition],
            edgecolor="none",
            alpha=0.72,
            zorder=2,
        )
        ax.errorbar(
            index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            color="black",
            fmt="none",
            capsize=4.0 * FONT_SCALE,
            elinewidth=1.1 * FONT_SCALE,
            capthick=1.1 * FONT_SCALE,
            zorder=4,
        )

    ax.set_xticks(
        range(2),
        [CONDITION_LABELS["latent_day2"], CONDITION_LABELS["return"]],
        rotation=0,
        ha="center",
    )
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel(claim["ylabel"])
    ax.set_title(claim["title"], loc="left", fontweight="bold", pad=2.0)
    ax.grid(False)

    # Bars are anchored at zero; only the direct-minus-other contrast goes negative.
    low_value = min(observed)
    high_value = max(observed)
    bottom = min(0.0, low_value - 0.06 * (high_value - low_value))
    ax.set_ylim(bottom, high_value + 0.06 * (high_value - bottom))
    if claim["claim_id"] == "ed4_pre_reward_direct_chain_specificity_check":
        ax.axhline(0.0, color=LEGACY_COLOR, linewidth=0.55, linestyle="--", zorder=3)


def significance_stars(q_value: float) -> str:
    """Main Figure 1 star thresholds, with an explicit mark when unsupported.

    Figure 1 leaves non-significant contrasts blank. Here ``n.s.`` is drawn instead,
    because two of the four ED3 claims are deliberately unsupported and a blank bracket
    would read as untested rather than as tested-and-null.
    """
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "n.s."


def draw_significance_bracket(
    ax: plt.Axes, x_left: float, x_right: float, q_value: float
) -> None:
    """Draw the Main Figure 1 bracket-and-star over a pairwise comparison."""
    low, high = ax.get_ylim()
    span = high - low
    y_base = low + 0.86 * span
    tick = 0.022 * span
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y_base, y_base + tick, y_base + tick, y_base],
        color=AXIS_COLOR,
        linewidth=0.7,
        clip_on=False,
        zorder=5,
    )
    stars = significance_stars(q_value)
    ax.text(
        (x_left + x_right) / 2.0,
        y_base + tick * 1.1,
        stars,
        ha="center",
        va="bottom",
        fontsize=BASE_FONT_SIZE if stars == "n.s." else MAIN_BASE_FONT_SIZE * FONT_SCALE * 1.15,
        fontweight="normal" if stars == "n.s." else "bold",
        color=AXIS_COLOR,
        clip_on=False,
        zorder=6,
    )


def draw_maze_decision_schematic(
    ax: plt.Axes, definition: pd.DataFrame
) -> dict[str, Any]:
    """Draw the six scored decision points on the 2D maze map.

    The reward path and its branch points are resolved from the maze geometry, so the
    drawn decision points are the actual junctions of the runs named in the maintained
    route-definition table rather than hand-placed marks.
    """
    definition = definition.sort_values("paper_depth").reset_index(drop=True)
    table_parents = [int(value) for value in definition["parent_node"]]
    table_children = [int(value) for value in definition["reward_child"]]
    runs = [table_parents[0]] + table_children
    if tuple(runs) != REWARD_PATH_RUNS:
        raise ValueError(f"Route definition does not match the reward path: {runs}")

    # Canonical main-figure maze schematic: same walls, gray blocks, home icon and water
    # drop. Only the coloured end-node quadrants and the long-bout path are suppressed.
    maze = get_panel_maze(6)
    panel_size = (SCHEMATIC_RECT_IN[2], SCHEMATIC_RECT_IN[3])
    draw_maze_outline(ax, maze, panel_size=panel_size)
    draw_gray_blocks(ax, panel_size=panel_size, blocks=GRAY_BLOCKS)
    draw_quadrant_overlays(ax, maze, panel_size=panel_size, quadrants=[])
    draw_maze_landmarks(ax)

    # The reward path is the concatenation of the runs; consecutive runs are adjacent.
    path_x: list[float] = []
    path_y: list[float] = []
    for run in runs:
        for cell in maze.ru[run]:
            path_x.append(float(maze.xc[cell]))
            path_y.append(float(maze.yc[cell]))
    ax.plot(
        path_x,
        path_y,
        color=PATH_COLOR,
        linewidth=1.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )

    # A decision point is the last cell of a parent run, where it branches.
    decisions: list[dict[str, Any]] = []
    for depth, run in enumerate(runs[:-1]):
        cell = maze.ru[run][-1]
        decisions.append(
            {
                "depth": depth,
                "run": run,
                "xy": [int(maze.xc[cell]), int(maze.yc[cell])],
                "reward_side": str(definition.loc[depth, "reward_side"]).upper(),
            }
        )
    ax.scatter(
        [d["xy"][0] for d in decisions],
        [d["xy"][1] for d in decisions],
        s=26 * SCHEMATIC_SCALE**2,
        facecolor="white",
        edgecolor=PATH_COLOR,
        linewidth=0.6 * SCHEMATIC_SCALE,
        zorder=5,
    )
    for decision in decisions:
        ax.text(
            decision["xy"][0],
            decision["xy"][1],
            str(decision["depth"]),
            ha="center",
            va="center",
            fontsize=TICK_LABEL_SIZE * 0.62 * SCHEMATIC_SCALE,
            color=PATH_COLOR,
            zorder=6,
        )

    # The water drop is the canonical landmark drawn by draw_maze_landmarks, which sits
    # outside the maze at the water port rather than on the terminal maze cell.
    landmarks = get_maze_schematic_landmarks()
    reward_xy = [float(v) for v in landmarks["reward"]["xy"]]
    reward_cell = maze.ru[runs[-1]][-1]
    terminal_xy = [int(maze.xc[reward_cell]), int(maze.yc[reward_cell])]

    ax.set_xlim(-2.2, 16.2)
    ax.set_ylim(15.4, -0.9)
    ax.set_aspect("equal")
    finish_panel(ax)
    return {
        "decisions": decisions,
        "reward_xy": reward_xy,
        "terminal_cell_xy": terminal_xy,
        "runs": runs,
    }


def draw_depth_panels(
    fig: plt.Figure, depth: pd.DataFrame
) -> tuple[list[plt.Axes], dict[str, dict[str, int]]]:
    axes: list[plt.Axes] = []
    support: dict[str, dict[str, int]] = {}
    for index, condition in enumerate(DEPTH_PANEL_ORDER):
        subset = depth.loc[depth["condition"].eq(condition)].sort_values("tree_depth")
        if len(subset) != 6:
            raise ValueError(
                f"Expected six depth bars for {condition}; found {len(subset)}"
            )
        ax = fig.add_axes(
            rect(
                DEPTH_PANEL_LEFTS[index],
                DEPTH_BOTTOM,
                DEPTH_PANEL_WIDTH,
                DEPTH_HEIGHT,
            )
        )
        color = DEPTH_PANEL_COLORS[condition]
        x = subset["tree_depth"].to_numpy(dtype=int)
        values = subset["mean_reward_correct"].to_numpy(dtype=float)
        errors = subset["sem_reward_correct"].to_numpy(dtype=float)

        ax.axhline(0.5, color="#A0A0A0", linewidth=0.5, linestyle="--", zorder=0)
        ax.bar(x, values, width=0.76, color=color, alpha=0.92, zorder=2)
        ax.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            color=AXIS_COLOR,
            capsize=1.4,
            elinewidth=0.6,
            capthick=0.6,
            zorder=3,
        )
        ax.set_xticks(range(6))
        ax.set_xlim(-0.55, 5.55)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.grid(False)
        ax.set_xlabel(
            DEPTH_PANEL_LABELS[condition],
            color=color,
            fontweight="bold",
            fontsize=AXIS_LABEL_SIZE,
            labelpad=3.0,
        )
        # Session/mouse counts are recorded in the manifest and belong in the caption.
        support[condition] = {
            "n_sessions": int(subset["n_sessions"].iloc[0]),
            "n_animals": int(subset["n_animals"].iloc[0]),
        }
        if index == 0:
            ax.set_ylabel("First choice toward reward", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        axes.append(ax)
    return axes, support


def build_figure(
    fluency: Any,
    rows: pd.DataFrame,
    statistics: pd.DataFrame,
    depth: pd.DataFrame,
    definition: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, Any]]:
    fluency.configure_style()
    # Override the ED4 renderer's own typography with the main-figure contract; this also
    # turns background grid lines off by default for every panel in this figure.
    configure_main_figure_typography()
    # Harmonise the fluency panels onto the Main Figure 1 condition palette and labels so
    # Return and Latent are one colour across the whole figure.
    fluency.COLOR["return"] = FIGURE1_RETURN_COLOR
    fluency.COLOR["latent"] = FIGURE1_LATENT_DAY2_COLOR
    fluency.CONDITION_LABEL["return"] = CONDITION_LABELS["return"]
    fluency.CONDITION_LABEL["latent"] = CONDITION_LABELS["latent_day2"]

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    fig.patch.set_facecolor("white")

    raw_stats = statistics.loc[statistics["scale"].eq("raw_seconds")].set_index(
        "claim_id"
    )
    letters = ("a", "b", "c", "d")
    claim_support: dict[str, Any] = {}
    # Figure 1's jitter seed; consumed in a fixed panel order so the render is reproducible.
    rng = np.random.default_rng(20260716)
    for index, claim in enumerate(fluency.CLAIMS):
        ax = fig.add_axes(
            rect(ROW1_PANEL_LEFTS[index], ROW1_BOTTOM, ROW1_PANEL_WIDTH, ROW1_HEIGHT)
        )
        subset = rows.loc[rows["claim_id"].eq(claim["claim_id"])]
        statistic = raw_stats.loc[claim["claim_id"]]
        draw_claim_bar_panel(ax, subset, claim, fluency, rng)

        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.16 * (high - low))
        draw_significance_bracket(
            ax, 0, 1, float(statistic["q_bh4_displayed_story"])
        )
        panel_letter(
            fig,
            ROW1_PANEL_LEFTS[index] - 0.38,
            ROW1_BOTTOM + ROW1_HEIGHT + 0.27,
            letters[index],
        )
        claim_support[claim["claim_id"]] = {
            "panel": letters[index],
            "title": claim["title"],
            "n_bars": sum(1 for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)),
            "n_rows": int(statistic["n_rows"]),
            "n_mice": int(statistic["n_mice"]),
            "beta": float(statistic["estimate_return_minus_latent"]),
            "p_cluster_two_sided": float(statistic["p_cluster_two_sided"]),
            "q_bh4_displayed_story": float(statistic["q_bh4_displayed_story"]),
        }

    schematic_ax = fig.add_axes(rect(*SCHEMATIC_RECT_IN))
    schematic = draw_maze_decision_schematic(schematic_ax, definition)
    # Keep the e and f letters on one baseline, with e aligned to the a letter above it.
    panel_letter(
        fig,
        ROW1_PANEL_LEFTS[0] - 0.38,
        DEPTH_BOTTOM + DEPTH_HEIGHT + 0.22,
        "e",
    )

    depth_axes, depth_support = draw_depth_panels(fig, depth)
    panel_letter(fig, DEPTH_PANEL_LEFTS[0] - 0.40, DEPTH_BOTTOM + DEPTH_HEIGHT + 0.22, "f")

    audit = {
        "panels": {
            "a-d": claim_support,
            "e": {
                "role": "scored decision points on the 2D maze map",
                "path_runs": schematic["runs"],
                "reward_node": schematic["runs"][-1],
                "reward_xy": schematic["reward_xy"],
                "terminal_cell_xy": schematic["terminal_cell_xy"],
                "decisions": schematic["decisions"],
                "sides": [d["reward_side"] for d in schematic["decisions"]],
            },
            "f": {"support": depth_support, "palette": DEPTH_PANEL_COLORS},
        },
        "condition_labels": CONDITION_LABELS,
        "fluency_return_colour": fluency.COLOR["return"],
        "n_axes": len(fig.axes),
        "grid_lines_visible": sum(
            1
            for axis in fig.axes
            for gridline in axis.get_xgridlines() + axis.get_ygridlines()
            if gridline.get_visible()
        ),
        "max_xticklabel_rotation": max(
            (label.get_rotation() for axis in fig.axes for label in axis.get_xticklabels()),
            default=0.0,
        ),
    }
    return fig, audit


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in FORMATS:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def validate(outputs: list[Path], audit: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    record("axes_count", audit["n_axes"] == 9, f"{audit['n_axes']} axes (4 fluency + 1 schematic + 4 depth)")
    record(
        "no_background_grid_lines",
        audit["grid_lines_visible"] == 0,
        f"{audit['grid_lines_visible']} visible gridlines across all axes",
    )
    record(
        "xticklabels_not_rotated",
        audit["max_xticklabel_rotation"] == 0.0,
        f"max x tick label rotation {audit['max_xticklabel_rotation']} deg",
    )

    claims = audit["panels"]["a-d"]
    expected = {
        "ed4_broad_pre_reward_explore": (-0.307587, 63, 20),
        "ed4_broad_post_reward_explore": (-0.082389, 65, 21),
        "ed4_pre_reward_direct_chain_specificity_check": (0.181756, 61, 20),
        "ed4_reward_approach_after_reward": (-0.186428, 65, 21),
    }
    for claim_id, (beta, n_rows, n_mice) in expected.items():
        got = claims[claim_id]
        ok = (
            abs(got["beta"] - beta) < 1e-5
            and got["n_rows"] == n_rows
            and got["n_mice"] == n_mice
        )
        record(
            f"claim_{claim_id}",
            ok,
            f"beta={got['beta']:.6f} n={got['n_rows']}/{got['n_mice']} q={got['q_bh4_displayed_story']:.3g}",
        )

    schematic = audit["panels"]["e"]
    record(
        "schematic_path_runs",
        list(schematic["path_runs"]) == [0, 1, 4, 10, 22, 46, 94],
        str(schematic["path_runs"]),
    )
    record(
        "schematic_reward_node_94",
        schematic["reward_node"] == 94,
        f"terminal run {schematic['reward_node']}",
    )
    record(
        "schematic_terminal_cell_is_node94_grid",
        schematic["terminal_cell_xy"] == [14, 6],
        f"run 94 terminal cell at grid {schematic['terminal_cell_xy']}",
    )
    record(
        "schematic_uses_canonical_water_drop",
        schematic["reward_xy"] == [15.18, 6.12],
        f"canonical water port landmark at {schematic['reward_xy']}",
    )
    record(
        "schematic_side_sequence",
        schematic["sides"] == ["LEFT", "RIGHT", "RIGHT", "LEFT", "RIGHT", "LEFT"],
        "-".join(s[0] for s in schematic["sides"]),
    )
    expected_decisions = [
        (0, [7, 7]),
        (1, [7, 3]),
        (2, [11, 3]),
        (3, [11, 5]),
        (4, [13, 5]),
        (5, [13, 6]),
    ]
    got_decisions = [(d["depth"], d["xy"]) for d in schematic["decisions"]]
    record(
        "schematic_six_decision_points_on_maze",
        got_decisions == expected_decisions,
        "; ".join(f"depth{d}@{xy}" for d, xy in got_decisions),
    )

    box_centre = SCHEMATIC_RECT_IN[0] + SCHEMATIC_RECT_IN[2] / 2.0
    column_centre = sum(SCHEMATIC_COLUMN_IN) / 2.0
    record(
        "schematic_centred_in_column",
        abs(box_centre - column_centre) < 1e-6,
        f"box centre {box_centre:.3f} in vs column centre {column_centre:.3f} in",
    )

    labels = audit["condition_labels"]
    record(
        "labels_drop_redundant_day1",
        labels["naive"] == "Naive day"
        and labels["return"] == "Return day"
        and not any(v.endswith(" 1") for v in labels.values()),
        json.dumps(labels),
    )
    record(
        "latent_label_uses_author_approved_style",
        labels["latent_day2"] == "Latent day2",
        labels["latent_day2"],
    )

    support = audit["panels"]["f"]["support"]
    expected_support = {
        "naive": (23, 23),
        "experienced_day2": (74, 24),
        "return": (56, 17),
        "latent_day2": (9, 9),
    }
    for condition, (sessions, animals) in expected_support.items():
        got = support[condition]
        ok = got["n_sessions"] == sessions and got["n_animals"] == animals
        record(
            f"support_{condition}",
            ok,
            f"{got['n_sessions']} sessions / {got['n_animals']} mice",
        )
    record(
        "fluency_panels_use_bar_plus_scatter",
        all(c["n_bars"] == 2 for c in claims.values()),
        "two condition bars per panel in a-d, Main Figure 1 grammar",
    )

    palette = audit["panels"]["f"]["palette"]
    record(
        "depth_palette_matches_main_figure1",
        palette["naive"] == FIGURE1_NAIVE_COLOR
        and palette["experienced_day2"] == FIGURE1_SECOND_DAY_COLOR
        and palette["latent_day2"] == FIGURE1_LATENT_DAY2_COLOR,
        json.dumps(palette),
    )
    record(
        "return_colour_softened_deliberately",
        palette["return"] == FIGURE1_RETURN_COLOR != FIGURE1_RETURN_COLOR_MAIN,
        f"{palette['return']} here vs {FIGURE1_RETURN_COLOR_MAIN} in Main Figure 1 — "
        "documented deviation, not drift",
    )
    record(
        "return_colour_consistent_across_panels",
        audit["fluency_return_colour"] == palette["return"],
        f"panels a-d use {audit['fluency_return_colour']}, panel f uses {palette['return']}",
    )

    for path in outputs:
        record(f"output_{path.suffix.lstrip('.')}", path.exists() and path.stat().st_size > 0, f"{path.name} {path.stat().st_size} bytes")

    svg = next(p for p in outputs if p.suffix == ".svg")
    text = svg.read_text(encoding="utf-8", errors="ignore")
    record("svg_editable_text", "<text" in text, "SVG contains live <text> elements")
    record("svg_declares_arial", "Arial" in text, "Arial declared in SVG")

    # Text the user asked to move into the caption must not survive in the figure.
    for phrase, label in (
        ("CR1", "cluster-P annotation"),
        ("BH4", "BH4 q annotation"),
        ("sessions /", "per-panel session/mouse counts"),
        ("Equal-animal", "mean/SEM note"),
        ("day 1", "redundant day-1 suffix"),
    ):
        record(
            f"removed_{phrase.strip().replace(' ', '_').replace('/', '')}",
            phrase not in text,
            f"{label} absent from SVG",
        )
    return checks


def run(output_dir: Path) -> dict[str, Any]:
    fluency = load_fluency_module()
    rows = pd.read_csv(FLUENCY_ROWS)
    statistics = pd.read_csv(FLUENCY_STATS)
    depth = pd.read_csv(DEPTH_TABLE)
    definition = pd.read_csv(ROUTE_DEFINITION)

    fig, audit = build_figure(fluency, rows, statistics, depth, definition)
    stem = output_dir / "ed03_reward_path_fluency_assembly"
    outputs = save_figure(fig, stem)
    plt.close(fig)

    checks = validate(outputs, audit)
    manifest = {
        "figure": "Extended Data Figure 3 (slide 4 of sup-figs-v2.pptx)",
        "layout_authority": "slide 4; Extended Data Figure 2 merged into Figure 1 shifts this slide to the Figure 3 role",
        "figure_size_inches": [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN],
        "sources": {
            "fluency_plot_rows": {"path": str(FLUENCY_ROWS.relative_to(REPO_ROOT)), "sha256": sha256(FLUENCY_ROWS)},
            "fluency_statistics": {"path": str(FLUENCY_STATS.relative_to(REPO_ROOT)), "sha256": sha256(FLUENCY_STATS)},
            "depth_small_multiples": {"path": str(DEPTH_TABLE.relative_to(REPO_ROOT)), "sha256": sha256(DEPTH_TABLE)},
            "route_definition": {"path": str(ROUTE_DEFINITION.relative_to(REPO_ROOT)), "sha256": sha256(ROUTE_DEFINITION)},
        },
        "slide_corrections": [
            "panels a-d redrawn from the manuscript-facing claim contract instead of the role=source QA overview crop",
            "panel e replaced the 99_legacy_audit_only first-turn maze schematic with the current first-visit six-decision definition",
            "panel f uses the maintained condition palette, which matches Main Figure 1; the slide palette is superseded",
        ],
        "audit": audit,
        "outputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "validation": checks,
        "validation_status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ED03_ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(checks).to_csv(output_dir / "ED03_VALIDATION.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args().output_dir)
    failed = [c for c in manifest["validation"] if c["status"] != "PASS"]
    print(json.dumps({"validation_status": manifest["validation_status"], "checks": len(manifest["validation"]), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
