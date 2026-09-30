"""Render legacy GRS slide-3 trajectories from recorded maze bout 0.

The poster trajectory assets stop at first reward. This slide-specific renderer
instead selects behavior bout 0 in full and retains nose coordinates. A bout is
a maze-presence interval, not an endpoint-validated behavioral trial; these
historical filenames are retained only for compatibility. Nose-keypoint jump
islands containing repeated implausible edges are omitted from the drawn
trajectory; no movement is imputed across them. Remaining segments are not
bridged across missing frames or speeds above the established 60 cm/s tracking
ceiling (approximately 0.63 maze-grid units per frame).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image

from ca_analysis.config import FPS, get_behav_wd
from ca_analysis.maze import NewMaze
from ca_analysis.trajectory import LoadTraj


SESSIONS = (
    {"nickname": "434RRday1", "display_title": "Day1", "slide": 3},
    {"nickname": "434RRday2", "display_title": "Day2", "slide": 3},
)
FIRST_BOUT_INDEX = 0
MAX_STEP_PER_FRAME_GRID = 0.63
JUMP_ISLAND_MAX_EDGE_GAP_FRAMES = 25
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig, stem: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    for suffix, kwargs in {
        ".png": {"dpi": dpi},
        ".svg": {},
        ".pdf": {},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


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


def _draw_trajectory(
    ax,
    data: pd.DataFrame,
    display_title: str,
    norm: Normalize,
) -> LineCollection:
    maze = NewMaze(6)
    ax.plot(maze.wa[:, 0], maze.wa[:, 1], color="black", linewidth=0.62, zorder=2)
    for x, y, width, height in GRAY_BLOCKS:
        ax.add_patch(Rectangle((x, y), width, height, linewidth=0.35,
                               edgecolor="#d8d8d8", facecolor="#d8d8d8", zorder=1))

    points = data[["x", "y"]].to_numpy(dtype=float)
    elapsed = data["elapsed_minutes"].to_numpy(dtype=float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    keep = data["draw_to_next"].iloc[:-1].astype(bool).to_numpy()
    collection = LineCollection(
        segments[keep], cmap=plt.get_cmap("viridis"), norm=norm,
        linewidths=0.45, alpha=0.86, rasterized=True, capstyle="round", joinstyle="round",
    )
    collection.set_array(0.5 * (elapsed[:-1] + elapsed[1:])[keep])
    ax.add_collection(collection)
    _draw_house_icon(ax)
    _draw_water_drop_icon(ax)
    ax.set_xlim(-1.85, 15.95)
    ax.set_ylim(15.0, -1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(display_title, fontsize=6.2, fontweight="normal", pad=2.5)
    return collection


def _format_elapsed_colorbar(cbar, max_minutes: float) -> None:
    cbar.outline.set_linewidth(0.45)
    cbar.set_ticks([0.0, float(max_minutes)])
    cbar.set_ticklabels(["0", f"{max_minutes:.1f}"])
    cbar.ax.tick_params(labelsize=5.4, width=0.45, length=1.5, pad=1.0)
    cbar.set_label("Elapsed time (min)", fontsize=6.0, labelpad=1.5)


def _render_panel(
    data: pd.DataFrame,
    display_title: str,
    stem: Path,
    dpi: int,
    shared_max_minutes: float,
) -> list[Path]:
    fig = plt.figure(figsize=(2.65, 3.05), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.065], hspace=0.19)
    ax = fig.add_subplot(grid[0, 0])
    norm = Normalize(0.0, float(shared_max_minutes))
    _draw_trajectory(ax, data, display_title, norm)

    cax = fig.add_subplot(grid[1, 0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                        cax=cax, orientation="horizontal")
    _format_elapsed_colorbar(cbar, shared_max_minutes)
    return _save_figure(fig, stem, dpi)


def _render_shared_pair(
    data_rows: list[pd.DataFrame],
    titles: list[str],
    stem: Path,
    dpi: int,
    shared_max_minutes: float,
) -> list[Path]:
    fig = plt.figure(figsize=(4.7, 2.35), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.075], hspace=0.20, wspace=0.08)
    norm = Normalize(0.0, float(shared_max_minutes))
    for col, (data, title) in enumerate(zip(data_rows, titles, strict=True)):
        _draw_trajectory(fig.add_subplot(grid[0, col]), data, title, norm)
    cax = fig.add_subplot(grid[1, :])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=cax,
        orientation="horizontal",
    )
    _format_elapsed_colorbar(cbar, shared_max_minutes)
    return _save_figure(fig, stem, dpi)


def _identify_nose_jump_islands(
    frames: np.ndarray,
    raw_nose_xy: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]], np.ndarray]:
    """Return a removal mask for nose islands containing repeated fast edges."""
    frame_gap = np.diff(frames)
    raw_step = np.linalg.norm(np.diff(raw_nose_xy, axis=0), axis=1)
    raw_step_per_frame = np.divide(
        raw_step,
        frame_gap,
        out=np.full(raw_step.shape, np.inf, dtype=float),
        where=frame_gap > 0,
    )
    high_edges = np.where((frame_gap == 1) & (raw_step_per_frame > MAX_STEP_PER_FRAME_GRID))[0]
    edge_groups: list[list[int]] = []
    if len(high_edges):
        group = [int(high_edges[0])]
        for edge in high_edges[1:]:
            edge = int(edge)
            if int(frames[edge] - frames[group[-1]]) <= JUMP_ISLAND_MAX_EDGE_GAP_FRAMES:
                group.append(edge)
            else:
                edge_groups.append(group)
                group = [edge]
        edge_groups.append(group)

    removed = np.zeros(len(frames), dtype=bool)
    islands: list[dict[str, int]] = []
    for group in edge_groups:
        if len(group) < 2:
            continue
        left_idx = int(group[0])
        right_idx = int(group[-1] + 1)
        if right_idx <= left_idx + 1 or right_idx >= len(frames):
            continue
        inner = np.arange(left_idx + 1, right_idx, dtype=int)
        removed[inner] = True
        islands.append(
            {
                "first_fast_edge_frame": int(frames[left_idx]),
                "last_fast_edge_frame": int(frames[right_idx]),
                "n_fast_edges": int(len(group)),
                "n_removed_points": int(len(inner)),
                "total_span_frames": int(frames[right_idx] - frames[left_idx]),
                "endpoint_distance_grid": float(
                    np.linalg.norm(raw_nose_xy[right_idx] - raw_nose_xy[left_idx])
                ),
            }
        )
    return removed, islands, raw_step_per_frame


def _selected_bout_data(
    nickname: str,
    display_title: str,
    selected_data_dir: Path,
    *,
    bout_index: int,
    allow_partial_selected_coverage: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return audited corrected-nose rows for one explicit trajectory bout."""
    source = selected_data_dir / f"{nickname}_nose_interp_data.csv"
    columns = ["frame", "x", "y", "bout"]
    data = pd.read_csv(source, usecols=columns).dropna(subset=["frame", "bout", "x", "y"])
    selected_bout = int(bout_index)
    available_bouts = set(data["bout"].astype(int).unique().tolist())
    if selected_bout not in available_bouts:
        raise ValueError(
            f"{nickname}: requested bout {selected_bout} is absent from corrected-nose data"
        )
    data = (
        data.loc[data["bout"].astype(int).eq(selected_bout)]
        .sort_values("frame")
        .reset_index(drop=True)
    )
    if len(data) < 2:
        raise ValueError(f"{nickname}: selected maze bout has fewer than two tracking rows")

    tf = LoadTraj(f"{nickname}-tf")
    if selected_bout < 0 or selected_bout >= len(tf.fr):
        raise ValueError(
            f"{nickname}: requested bout {selected_bout} is outside trajectory bounds"
        )
    expected_start, expected_end = map(int, tf.fr[selected_bout])
    first_frame = int(data["frame"].iloc[0])
    last_frame = int(data["frame"].iloc[-1])
    start_offset_frames = int(first_frame - expected_start)
    end_offset_frames = int(expected_end - last_frame)
    exact_selected_coverage = bool(
        first_frame == expected_start and last_frame in {expected_end - 1, expected_end}
    )
    partial_selected_coverage_valid = bool(
        expected_start <= first_frame <= expected_end
        and expected_start <= last_frame <= expected_end
        and first_frame < last_frame
    )
    if not exact_selected_coverage and not (
        allow_partial_selected_coverage and partial_selected_coverage_valid
    ):
        raise ValueError(
            f"{nickname}: selected-data bout {selected_bout} {first_frame}-{last_frame} does not match "
            f"trajectory bounds {expected_start}-{expected_end}"
        )

    frames = data["frame"].to_numpy(dtype=int)
    frame_gap = np.diff(frames)
    raw_nose_xy = np.column_stack(
        (
            -0.5 + 15.0 * data["x"].to_numpy(dtype=float),
            -0.5 + 15.0 * data["y"].to_numpy(dtype=float),
        )
    )
    removed_points, jump_islands, raw_nose_step_per_frame = _identify_nose_jump_islands(
        frames, raw_nose_xy
    )
    nose_xy = raw_nose_xy.copy()
    draw_to_next = np.zeros(len(data), dtype=bool)
    visible_endpoints = ~(removed_points[:-1] | removed_points[1:])
    draw_to_next[:-1] = (
        (frame_gap == 1)
        & np.isfinite(raw_nose_step_per_frame)
        & (raw_nose_step_per_frame <= MAX_STEP_PER_FRAME_GRID)
        & visible_endpoints
    )

    elapsed_seconds = (frames - frames[0]) / float(FPS)
    duration_seconds = float(elapsed_seconds[-1])
    progress = elapsed_seconds / duration_seconds
    plot_data = pd.DataFrame(
        {
            "row_idx": 0,
            "col_idx": 0,
            "nickname": nickname,
            "display_title": display_title,
            "bout_idx": selected_bout,
            "point_idx": np.arange(len(data), dtype=int),
            "frame": frames,
            "x": nose_xy[:, 0],
            "y": nose_xy[:, 1],
            "raw_nose_x": raw_nose_xy[:, 0],
            "raw_nose_y": raw_nose_xy[:, 1],
            "is_jump_island_removed": removed_points,
            "progress": progress,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "draw_to_next": draw_to_next,
            "position_source": "nose_x_y_raw_with_repeated_jump_islands_masked",
            "trajectory_source": "selected_data_nose_interp_table_nose_columns",
            "trajectory_path": str(source),
            "clip_mode": "selected_full_bout",
            "selected_bout_index": selected_bout,
            "trajectory_tf_selected_bout_start": expected_start,
            "trajectory_tf_selected_bout_end": expected_end,
            "selected_start_offset_frames": start_offset_frames,
            "selected_end_offset_frames": end_offset_frames,
            "exact_selected_coverage": exact_selected_coverage,
            "max_step_per_frame_grid": MAX_STEP_PER_FRAME_GRID,
            "tracking_correction": "omit_repeated_fast_edge_islands_then_split_gaps_or_gt60cm_s",
        }
    )
    summary = {
        "nickname": nickname,
        "display_title": display_title,
        "selected_bout_index": selected_bout,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "trajectory_tf_selected_bout_start": expected_start,
        "trajectory_tf_selected_bout_end": expected_end,
        "selected_start_offset_frames": start_offset_frames,
        "selected_end_offset_frames": end_offset_frames,
        "exact_selected_coverage": exact_selected_coverage,
        "allow_partial_selected_coverage": bool(allow_partial_selected_coverage),
        "n_points": int(len(data)),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60.0,
        "position_source": "nose_x_y_raw_with_repeated_jump_islands_masked",
        "max_step_per_frame_grid": MAX_STEP_PER_FRAME_GRID,
        "jump_island_max_edge_gap_frames": JUMP_ISLAND_MAX_EDGE_GAP_FRAMES,
        "raw_nose_segments_above_threshold": int(
            np.sum(raw_nose_step_per_frame > MAX_STEP_PER_FRAME_GRID)
        ),
        "n_jump_islands_removed": int(len(jump_islands)),
        "n_jump_island_points_removed": int(np.sum(removed_points)),
        "jump_islands": jump_islands,
        "nonconsecutive_frame_segments": int(np.sum(frame_gap != 1)),
        "suppressed_draw_segments": int(np.sum(~draw_to_next[:-1])),
        "source_csv": str(source),
        "source_csv_sha256": _sha256(source),
    }
    return plot_data, summary


def _first_trial_data(
    nickname: str,
    display_title: str,
    selected_data_dir: Path,
    *,
    allow_partial_selected_coverage: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Legacy wrapper selecting recorded bout 0 without endpoint validation."""
    plot_data, summary = _selected_bout_data(
        nickname,
        display_title,
        selected_data_dir,
        bout_index=FIRST_BOUT_INDEX,
        allow_partial_selected_coverage=allow_partial_selected_coverage,
    )
    plot_data = plot_data.copy()
    plot_data["trajectory_tf_first_bout_start"] = plot_data[
        "trajectory_tf_selected_bout_start"
    ]
    plot_data["trajectory_tf_first_bout_end"] = plot_data[
        "trajectory_tf_selected_bout_end"
    ]
    summary = dict(summary)
    summary["trajectory_tf_first_bout_start"] = summary[
        "trajectory_tf_selected_bout_start"
    ]
    summary["trajectory_tf_first_bout_end"] = summary[
        "trajectory_tf_selected_bout_end"
    ]
    return plot_data, summary


def render_first_trial_panels(selected_data_dir: Path, output_dir: Path, dpi: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    outputs: list[str] = []
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    for spec in SESSIONS:
        data, summary = _first_trial_data(
            spec["nickname"], spec["display_title"], selected_data_dir
        )
        all_data.append(data)
        summaries.append(summary)

    shared_max_minutes = max(float(row["duration_minutes"]) for row in summaries)
    for spec, data in zip(SESSIONS, all_data, strict=True):
        stem = output_dir / f"grs2026_{spec['display_title'].lower()}_first_maze_trial_nose_jump_masked"
        saved = _render_panel(
            data,
            spec["display_title"],
            stem,
            dpi,
            shared_max_minutes,
        )
        outputs.extend(str(path) for path in saved)

    shared_stem = output_dir / "grs2026_day1_day2_first_maze_trial_nose_shared_elapsed_time"
    outputs.extend(
        str(path)
        for path in _render_shared_pair(
            all_data,
            [str(spec["display_title"]) for spec in SESSIONS],
            shared_stem,
            dpi,
            shared_max_minutes,
        )
    )

    plot_data = pd.concat(all_data, ignore_index=True)
    plot_data_path = output_dir / "grs2026_first_maze_trial_trajectory_plot_data.csv"
    summary_path = output_dir / "grs2026_first_maze_trial_trajectory_summary.csv"
    plot_data.to_csv(plot_data_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    pngs = [Path(path) for path in outputs if Path(path).suffix.lower() == ".png"]
    validation = []
    for path in pngs:
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
        validation.append(
            {
                "path": str(path),
                "nonblank": bool(max(high - low for low, high in extrema) > 1),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "figure_role": "slide_3_recorded_bout_0_examples_legacy",
        "backend": "python_matplotlib",
        "selection": "recorded_bout_0_full_maze_presence_interval",
        "terminology_warning": (
            "Traj.fr bout 0 is not endpoint-validated and must not be called a "
            "behavioral trial"
        ),
        "position_source": "nose_x_y_raw_with_repeated_jump_islands_masked",
        "tracking_rule": {
            "max_step_per_frame_grid": MAX_STEP_PER_FRAME_GRID,
            "equivalent_speed_ceiling": "approximately_60_cm_s",
            "jump_island_max_edge_gap_frames": JUMP_ISLAND_MAX_EDGE_GAP_FRAMES,
            "jump_island_min_fast_edges": 2,
            "jump_island_action": "omit_all_segments_touching_internal_island_points",
            "imputed_movement": False,
            "bridge_nonconsecutive_frames": False,
        },
        "progress_color": "shared_actual_elapsed_minutes_across_day1_day2",
        "shared_elapsed_time_max_minutes": shared_max_minutes,
        "plot_data_csv": str(plot_data_path),
        "summary_csv": str(summary_path),
        "summaries": summaries,
        "outputs": outputs,
        "validation": validation,
    }
    manifest_path = output_dir / "grs2026_first_maze_trial_trajectory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not all(row["nonblank"] for row in validation):
        raise RuntimeError("At least one first-trial trajectory panel is blank")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-data-dir",
        type=Path,
        default=Path(get_behav_wd()) / "selected_data",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    manifest = render_first_trial_panels(args.selected_data_dir, args.output_dir, int(args.dpi))
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": manifest["summaries"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
