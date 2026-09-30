"""Recover Figure 1's complete rewarded timeline events from saved trajectories.

The accepted Round 12 artwork includes nodes 63--126, but its archived rewarded
timeline CSV includes only 95--126. This read-only helper repeats the accepted
event/clock extraction and checks every point against the accepted SVG. It does
not segment trajectories, recompute behavioral measures, or write artifacts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ca_analysis.trajectory import LoadTraj, TimeInMaze


DEFAULT_TRAJECTORY_DIR = Path("D:/combineddata/mm-trajectories")
REWARDED_SESSIONS = (
    "434RRday1",
    "434-3day1",
    "434RRday2",
    "434-6day1",
)
EXPECTED_EVENT_COUNTS = (1466, 738, 1337, 839)
ACCEPTED_QUADRANT_COLORS = ("#e0b27d", "#9fbf96", "#a38fbf", "#cc9a87")
SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _svg_events(by_id: dict, collection_id: str) -> tuple[np.ndarray, list[str]]:
    """Invert the accepted native axes, without using assembly transforms."""
    collection = by_id[collection_id]
    clipped = next(
        node
        for node in collection.findall(".//svg:g", SVG_NAMESPACE)
        if node.get("clip-path")
    )
    clip_reference = re.fullmatch(r"url\(#([^()]+)\)", clipped.attrib["clip-path"])
    if clip_reference is None:
        raise ValueError(f"Unsupported clip reference in {collection_id}")
    rectangle = by_id[clip_reference.group(1)].find("svg:rect", SVG_NAMESPACE)
    if rectangle is None:
        raise ValueError(f"Missing axes rectangle in {collection_id}")
    x, y, width, height = (float(rectangle.attrib[k]) for k in ("x", "y", "width", "height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid axes rectangle in {collection_id}")
    points = collection.findall(".//svg:use", SVG_NAMESPACE)
    # The accepted renderer fixes xlim=(0,6500), ylim=(61,128).
    values = np.asarray(
        [
            [
                (float(point.attrib["x"]) - x) / width * 6500,
                128 - (float(point.attrib["y"]) - y) / height * 67,
            ]
            for point in points
        ],
        dtype=float,
    )
    colors = []
    for point in points:
        fill = re.search(r"(?:^|;)\s*fill:\s*(#[0-9a-fA-F]{6})(?:;|$)", point.get("style", ""))
        if fill is None:
            raise ValueError(f"Missing scatter fill in {collection_id}")
        colors.append(fill.group(1).lower())
    return values, colors


def build_complete_rewarded_timelines(
    record: Path,
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Return the four complete event tables and source/parity evidence.

    The returned columns match the record CSV: nickname, time_in_maze_s,
    endnode. Callers can append its unchanged latent-session rows and save the
    result into their own draft directory. All inputs remain read-only.
    """
    record = Path(record).resolve()
    trajectory_dir = Path(trajectory_dir).resolve()
    svg_path = record / "figure1_version_record.svg"
    archived_path = record / "plot_ready/figure1_endnode_timelines.csv"
    trajectory_paths = [trajectory_dir / f"{session}-tf" for session in REWARDED_SESSIONS]
    source_paths = [svg_path, archived_path, *trajectory_paths]
    source_hashes = {str(path): _sha256(path) for path in source_paths}
    root = ET.parse(svg_path).getroot()
    by_id = {element.attrib["id"]: element for element in root.iter() if element.get("id")}
    archived = pd.read_csv(archived_path, float_precision="round_trip")
    frames = []
    audits = []
    for index, (session, expected_count, trajectory_path) in enumerate(
        zip(REWARDED_SESSIONS, EXPECTED_EVENT_COUNTS, trajectory_paths, strict=True),
        start=1,
    ):
        trajectory = LoadTraj(trajectory_path)
        values = []
        # This is the accepted extract_endnode_timeline selection and clock.
        for bout_index in range(len(trajectory.no)):
            bout = np.asarray(trajectory.no[bout_index])
            if bout.ndim != 2 or bout.shape[1] < 2:
                continue
            nodes = bout[:, 0]
            absolute_times = bout[:, 1] + trajectory.fr[bout_index][0]
            maze_times = [TimeInMaze(time, trajectory) for time in absolute_times]
            for node, maze_time in zip(nodes, maze_times, strict=True):
                if 63 <= node <= 126:
                    values.append((float(maze_time), int(node)))
        events = np.asarray(values, dtype=float)
        if len(events) != expected_count:
            raise AssertionError(f"{session}: expected {expected_count} complete events, got {len(events)}")
        collection_id = f"c_PathCollection_{index}"
        svg_values, svg_colors = _svg_events(by_id, collection_id)
        if events.shape != svg_values.shape:
            raise AssertionError(f"{session}: extracted/SVG shapes differ: {events.shape}, {svg_values.shape}")
        errors = np.max(np.abs(events - svg_values), axis=0)
        # SVG coordinates were serialized to six decimals in native points.
        if errors[0] > 1e-4 or errors[1] > 1e-6:
            raise AssertionError(f"{session}: accepted SVG parity failed: {errors.tolist()}")
        expected_colors = [ACCEPTED_QUADRANT_COLORS[(int(node) - 63) // 16] for node in events[:, 1]]
        if expected_colors != svg_colors:
            raise AssertionError(f"{session}: accepted SVG color/order parity failed")
        archived_values = archived.loc[
            archived.nickname.eq(session), ["time_in_maze_s", "endnode"]
        ].to_numpy(dtype=float)
        retained_values = events[events[:, 1] >= 95]
        if not np.array_equal(retained_values, archived_values):
            raise AssertionError(f"{session}: archived 95--126 subset parity failed")
        frame = pd.DataFrame(events, columns=["time_in_maze_s", "endnode"])
        frame["endnode"] = frame.endnode.astype(int)
        frame.insert(0, "nickname", session)
        frames.append(frame)
        audits.append(
            {
                "session": session,
                "trajectory_path": str(trajectory_path),
                "svg_collection_id": collection_id,
                "complete_event_count": len(events),
                "archived_event_count": len(archived_values),
                "restored_63_to_94_event_count": int((events[:, 1] < 95).sum()),
                "node_range": [int(events[:, 1].min()), int(events[:, 1].max())],
                "svg_event_count_and_order_match": True,
                "max_svg_time_error_seconds": float(errors[0]),
                "max_svg_node_error": float(errors[1]),
                "svg_quadrant_color_match": True,
                "archived_95_to_126_subset_exact_match": True,
            }
        )
    if any(_sha256(path) != source_hashes[str(path)] for path in source_paths):
        raise AssertionError("A timeline source changed during extraction")
    audit = {
        "status": "PASS",
        "source_hashes": source_hashes,
        "unchanged_source_hashes_verified": True,
        "known_archive_defect": "Rewarded CSV rows were extracted with lower_threshold=95; accepted SVG uses 63--126.",
        "extraction": "Original tf.no event order; retain nodes 63--126; absolute frame is event offset plus tf.fr[bout,0]; TimeInMaze converts to cumulative maze seconds.",
        "statistical_unit": "Illustrative end-node entry; no statistical estimation performed.",
        "reward_node": 94,
        "display_limits": {"time_in_maze_seconds": [0, 6500], "endnode": [61, 128]},
        "svg_serialization_tolerances": {"time_seconds": 1e-4, "node": 1e-6},
        "sessions": audits,
    }
    return pd.concat(frames, ignore_index=True), audit
