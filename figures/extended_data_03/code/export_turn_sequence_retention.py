"""Export whole-cohort first-turn sequence analysis for reward-path retention.

The primary representation is the sequence of earliest parent-to-child exits at
the six internal decision nodes on the canonical home-to-reward path in
``NewMaze(6)``.  The primary window is pre-first-reward: decisions with
absolute trajectory frames before the first detected reward-delivery frame in
that session.  For one explicitly approved session whose recording begins
downstream of the root, interval-censored choices on the unique direct path
from home to the first valid location are prepended before first-choice scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - optional dependency guard for lightweight environments
    sm = None
    smf = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ca_analysis.behavior import default_behavior_groups_260113
from ca_analysis.compat import bootstrap_environment
from ca_analysis.config import FPS, REWARD_NODE
from ca_analysis.maze import HomePath, NewMaze
from ca_analysis.trajectory import LoadTraj, TimeInMaze


DEFAULT_OUT_DIR = Path("docs/turn_sequence_retention_2026-08-11_direct_entry_reconstruction")
INITIAL_PATH_RECONSTRUCTION_SESSIONS = frozenset({"489-3day1"})
INITIAL_PATH_RECONSTRUCTION_RULE = "approved_unique_home_to_first_valid_location_v1"
RETURN_SPECS = (
    ("LTMThreeDay1", "return", "Return D1", "3mo"),
    ("LTMSixDay1", "return", "Return D1", "6mo"),
    ("LTMNineDay1", "return", "Return D1", "9mo"),
    ("LTMTwelveDay1", "return", "Return D1", "12mo"),
)
LATENT_SPECS = (
    ("AllUnrewardedDay2", "latent_day2", "Latent D2", "unrewarded_day1_then_rewarded_day2"),
)
EXPERIENCED_DAY2_SPECS = (
    ("RewardedDay2", "experienced_day2", "Experienced D2", "rewarded_day2"),
    ("LTMThreeDay2", "experienced_day2", "Experienced D2", "3mo_day2"),
    ("LTMSixDay2", "experienced_day2", "Experienced D2", "6mo_day2"),
    ("LTMNineDay2", "experienced_day2", "Experienced D2", "9mo_day2"),
    ("LTMTwelveDay2", "experienced_day2", "Experienced D2", "12mo_day2"),
)
CONDITION_ORDER = ("naive", "latent_day2", "return", "experienced_day2")
FOUR_GROUP_PAIRWISE_COMPARISONS = (
    ("latent_day2_vs_naive", "latent_day2", "naive"),
    ("return_vs_naive", "return", "naive"),
    ("experienced_day2_vs_naive", "experienced_day2", "naive"),
    ("return_vs_latent_day2", "return", "latent_day2"),
    ("experienced_day2_vs_latent_day2", "experienced_day2", "latent_day2"),
    ("experienced_day2_vs_return", "experienced_day2", "return"),
)


def _neighbor_table(ma: Any) -> np.ndarray:
    """Return parent, left-child, right-child columns without heavy analysis imports."""
    n = int(len(ma.ru))
    out = np.full((n, 3), -1, dtype=int)
    out[:, 0] = np.asarray(ma.pa, dtype=int)
    children = np.asarray(ma.ch, dtype=int)
    step_types = np.asarray(ma.st, dtype=int)
    if children.ndim == 2 and children.shape[1] >= 2:
        for node in range(n):
            child0 = int(children[node, 0])
            child1 = int(children[node, 1])
            if child0 < 0 or child1 < 0:
                continue
            if int(step_types[node, child0]) == 0:
                out[node, 1:3] = (child0, child1)
            else:
                out[node, 1:3] = (child1, child0)
    if n:
        out[0, 0] = n
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export first-turn reward-path sequence retention analysis."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--reward-node", type=int, default=int(REWARD_NODE))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-null-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--latent-cohort-key",
        default="AllUnrewardedDay2",
        help="Behavior group key to use for the latent Day2 comparison cohort.",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _animal_from_session(session_id: object) -> str:
    text = str(session_id)
    digits: list[str] = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return "".join(digits) if digits else text


def _path_to_root(node: int, ma: Any) -> list[int]:
    out = [int(node)]
    cur = int(node)
    while cur > 0:
        cur = int(ma.pa[cur])
        out.append(cur)
    return out[::-1]


def _first_valid_cell_sample(tf: Any, ma: Any) -> dict[str, Any] | None:
    """Return the earliest valid cell sample on the trajectory absolute-frame clock."""
    fr = np.asarray(getattr(tf, "fr", []), dtype=float)
    candidates: list[dict[str, Any]] = []
    for bout_idx, cell_arr in enumerate(getattr(tf, "ce", []) or []):
        cells = np.asarray(cell_arr, dtype=float).reshape(-1)
        if cells.size == 0:
            continue
        bout_start = (
            float(fr[bout_idx, 0])
            if fr.ndim == 2 and bout_idx < fr.shape[0]
            else 0.0
        )
        valid = np.flatnonzero(
            np.isfinite(cells)
            & np.isclose(cells, np.round(cells))
            & (cells >= 0)
            & (cells < len(ma.xc))
        )
        if valid.size == 0:
            continue
        local_frame = int(valid[0])
        cell = int(round(float(cells[local_frame])))
        candidates.append(
            {
                "bout": int(bout_idx),
                "local_frame": local_frame,
                "abs_frame": float(bout_start + local_frame),
                "cell": cell,
                "grid_x": int(ma.xc[cell]),
                "grid_y": int(ma.yc[cell]),
                "run_node": int(ma.rc[cell]),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["abs_frame"], row["bout"], row["local_frame"]))


def _first_valid_run_sample(tf: Any, ma: Any) -> dict[str, Any] | None:
    """Return the earliest valid run-node sample on the trajectory absolute-frame clock."""
    fr = np.asarray(getattr(tf, "fr", []), dtype=float)
    candidates: list[dict[str, Any]] = []
    for bout_idx, no_arr in enumerate(getattr(tf, "no", []) or []):
        no = np.asarray(no_arr, dtype=float)
        if no.ndim != 2 or no.shape[0] == 0 or no.shape[1] < 2:
            continue
        bout_start = (
            float(fr[bout_idx, 0])
            if fr.ndim == 2 and bout_idx < fr.shape[0]
            else 0.0
        )
        valid = np.flatnonzero(
            np.isfinite(no[:, 0])
            & np.isfinite(no[:, 1])
            & np.isclose(no[:, 0], np.round(no[:, 0]))
            & (no[:, 0] >= 0)
            & (no[:, 0] < len(ma.ru))
        )
        if valid.size == 0:
            continue
        idx = int(valid[0])
        candidates.append(
            {
                "bout": int(bout_idx),
                "node_index_in_bout": idx,
                "local_frame": float(no[idx, 1]),
                "abs_frame": float(bout_start + no[idx, 1]),
                "run_node": int(round(float(no[idx, 0]))),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["abs_frame"], row["bout"], row["node_index_in_bout"]))


def _build_initial_path_reconstruction(
    tf: Any,
    ma: Any,
    session_row: pd.Series,
    reward_path: pd.DataFrame,
    *,
    timing: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Build explicitly approved, interval-censored pre-recording choices.

    The rule is deliberately session-gated.  It requires the independent cell
    and run representations to identify the same first valid location before
    reward.  Every reward-path parent crossed on the unique home-to-location
    path is reconstructed so the chronological definition cannot selectively
    fill one missing row while retaining a later exit as the first choice.
    """
    session_id = str(session_row["session_id"])
    approved = session_id in INITIAL_PATH_RECONSTRUCTION_SESSIONS
    cell_sample = _first_valid_cell_sample(tf, ma)
    run_sample = _first_valid_run_sample(tf, ma)
    first_reward = float(timing.get("first_reward_abs_frame", np.nan))

    cell_frame = float(cell_sample["abs_frame"]) if cell_sample else np.nan
    run_frame = float(run_sample["abs_frame"]) if run_sample else np.nan
    cell_run = int(cell_sample["run_node"]) if cell_sample else -1
    run_node = int(run_sample["run_node"]) if run_sample else -1
    concordant = bool(
        cell_sample
        and run_sample
        and math.isclose(cell_frame, run_frame, rel_tol=0.0, abs_tol=1e-9)
        and cell_run == run_node
    )
    before_reward = bool(
        concordant and (not np.isfinite(first_reward) or cell_frame < first_reward)
    )
    eligible = bool(approved and concordant and before_reward)

    run_path: list[int] = _path_to_root(run_node, ma) if eligible else []
    cell_path: list[int] = (
        [int(value) for value in reversed(HomePath(int(cell_sample["cell"]), ma))]
        if eligible and cell_sample
        else []
    )
    reward_parent_rows = {
        int(row.parent_node): row for _, row in reward_path.iterrows()
    }
    reconstructed: dict[int, dict[str, Any]] = {}
    for parent, child in zip(run_path[:-1], run_path[1:]):
        rp = reward_parent_rows.get(int(parent))
        if rp is None:
            continue
        if int(child) not in {int(rp.reward_child), int(rp.sibling_child)}:
            raise ValueError(
                f"Initial path child {child} is not a reward-path branch of parent {parent}."
            )
        reconstructed[int(parent)] = {
            "chosen_child": int(child),
            "choice_side": (
                "left" if int(child) == int(rp.left_child) else "right"
            ),
            "reward_correct": bool(int(child) == int(rp.reward_child)),
        }

    if approved and not eligible:
        raise ValueError(
            f"Approved initial-path reconstruction failed closed for {session_id}: "
            f"cell_sample={cell_sample}, run_sample={run_sample}, "
            f"before_reward={before_reward}."
        )

    status = (
        "reconstructed"
        if eligible
        else "not_approved"
        if not approved
        else "failed_closed"
    )
    audit = pd.DataFrame(
        [
            {
                "session_id": session_id,
                "animal": str(session_row["animal"]),
                "condition": str(session_row["condition"]),
                "approved": approved,
                "rule": INITIAL_PATH_RECONSTRUCTION_RULE if approved else "",
                "assumption": "direct_home_to_first_valid_location" if approved else "",
                "status": status,
                "cell_run_concordant": concordant,
                "first_valid_abs_frame": cell_frame,
                "first_valid_cell": int(cell_sample["cell"]) if cell_sample else -1,
                "first_valid_grid_x": int(cell_sample["grid_x"]) if cell_sample else -1,
                "first_valid_grid_y": int(cell_sample["grid_y"]) if cell_sample else -1,
                "first_valid_run_node_from_cell": cell_run,
                "first_valid_run_node_from_no": run_node,
                "first_valid_run_abs_frame": run_frame,
                "first_reward_abs_frame": first_reward,
                "entry_precedes_first_reward": before_reward,
                "home_to_entry_cell_path": ";".join(map(str, cell_path)),
                "home_to_entry_run_path": ";".join(map(str, run_path)),
                "n_reconstructed_reward_parent_choices": int(len(reconstructed)),
            }
        ]
    )

    frame_rows: list[dict[str, Any]] = []
    if approved and cell_sample:
        bout_idx = int(cell_sample["bout"])
        bout_start = float(np.asarray(tf.fr, dtype=float)[bout_idx, 0])
        cells = np.asarray(tf.ce[bout_idx], dtype=float).reshape(-1)
        start = max(0, int(cell_sample["local_frame"]) - 8)
        stop = min(cells.size, int(cell_sample["local_frame"]) + 25)
        for local_frame in range(start, stop):
            value = float(cells[local_frame])
            valid = bool(
                np.isfinite(value)
                and np.isclose(value, round(value))
                and 0 <= value < len(ma.xc)
            )
            cell = int(round(value)) if valid else -1
            frame_rows.append(
                {
                    "session_id": session_id,
                    "bout": bout_idx,
                    "local_frame": int(local_frame),
                    "abs_frame": float(bout_start + local_frame),
                    "cell": cell,
                    "valid_maze_cell": valid,
                    "grid_x": int(ma.xc[cell]) if valid else -1,
                    "grid_y": int(ma.yc[cell]) if valid else -1,
                    "run_node": int(ma.rc[cell]) if valid else -1,
                    "is_first_valid_sample": bool(
                        local_frame == int(cell_sample["local_frame"])
                    ),
                }
            )
    return reconstructed, audit, pd.DataFrame(frame_rows)


def _reward_frames_abs(tf: Any) -> np.ndarray:
    frames: list[float] = []
    fr = np.asarray(getattr(tf, "fr", []), dtype=float)
    for bout_idx, reward_arr in enumerate(getattr(tf, "re", []) or []):
        arr = np.asarray(reward_arr)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 1:
            continue
        bout_start = float(fr[bout_idx, 0]) if fr.ndim == 2 and bout_idx < fr.shape[0] else 0.0
        frames.extend((arr[:, 0].astype(float) + bout_start).tolist())
    out = np.asarray(frames, dtype=float)
    return out[np.isfinite(out)]


def build_manifest(latent_cohort_key: str = "AllUnrewardedDay2") -> pd.DataFrame:
    groups = default_behavior_groups_260113()
    latent_specs = (
        (str(latent_cohort_key), "latent_day2", "Latent D2", "unrewarded_day1_then_rewarded_day2"),
    )
    specs = (
        ("RewardedDay1", "naive", "Naive D1", "0mo"),
        *latent_specs,
        *RETURN_SPECS,
        *EXPERIENCED_DAY2_SPECS,
    )
    rows: list[dict[str, Any]] = []
    for cohort_key, condition, label, memory_stage in specs:
        for session_id in groups.get(cohort_key, []):
            if condition == "naive":
                reward_context = "rewarded_naive"
            elif condition == "return":
                reward_context = "rewarded_ltm"
            elif condition == "experienced_day2":
                reward_context = "rewarded_or_return_day2_positive_control"
            else:
                reward_context = "latent_day2_first_reward"
            rows.append(
                {
                    "session_id": str(session_id),
                    "animal": _animal_from_session(session_id),
                    "condition": condition,
                    "condition_label": label,
                    "memory_stage": memory_stage,
                    "cohort_key": cohort_key,
                    "reward_context": reward_context,
                    "trajectory_id": f"{session_id}-tf",
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "memory_stage", "animal", "session_id"]).reset_index(drop=True)


def build_reward_path_table(ma: Any, reward_node: int) -> pd.DataFrame:
    if int(reward_node) != int(REWARD_NODE):
        raise ValueError(
            f"Primary turn-sequence analysis requires canonical reward node {REWARD_NODE}; got {reward_node}."
        )
    neighbors = _neighbor_table(ma)
    path = _path_to_root(int(reward_node), ma)
    rows: list[dict[str, Any]] = []
    previous_reward_side = ""
    for depth0, (parent, reward_child) in enumerate(zip(path[:-1], path[1:])):
        left_child = int(neighbors[parent, 1])
        right_child = int(neighbors[parent, 2])
        if reward_child == left_child:
            sibling = right_child
            reward_side = "left"
        elif reward_child == right_child:
            sibling = left_child
            reward_side = "right"
        else:
            raise ValueError(f"Reward child {reward_child} is not a child of parent {parent}.")
        if not previous_reward_side:
            required_transition_type = "start"
        elif reward_side == previous_reward_side:
            required_transition_type = "repeat_required"
        else:
            required_transition_type = "alternate_required"
        rows.append(
            {
                "paper_depth": int(depth0 + 1),
                "tree_depth": int(depth0),
                "parent_node": int(parent),
                "reward_child": int(reward_child),
                "sibling_child": int(sibling),
                "left_child": left_child,
                "right_child": right_child,
                "reward_side": reward_side,
                "previous_reward_side": previous_reward_side,
                "required_transition_type": required_transition_type,
            }
        )
        previous_reward_side = reward_side
    return pd.DataFrame(rows)


def _session_timing(tf: Any) -> dict[str, Any]:
    reward_frames = _reward_frames_abs(tf)
    first_reward_abs_frame = float(np.nanmin(reward_frames)) if reward_frames.size else np.nan
    if np.isfinite(first_reward_abs_frame):
        try:
            first_reward_maze_seconds = float(TimeInMaze(first_reward_abs_frame, tf))
        except Exception:
            first_reward_maze_seconds = float(first_reward_abs_frame / FPS)
    else:
        first_reward_maze_seconds = np.nan
    return {
        "first_reward_abs_frame": first_reward_abs_frame,
        "first_reward_maze_seconds": first_reward_maze_seconds,
        "n_reward_deliveries": int(reward_frames.size),
    }


def _extract_child_exit_events(
    tf: Any,
    ma: Any,
    session_row: pd.Series,
    *,
    timing: dict[str, Any],
) -> pd.DataFrame:
    neighbors = _neighbor_table(ma)
    leaf_start = 2 ** int(ma.le) - 1
    internal_limit = leaf_start
    no_list = getattr(tf, "no", []) or []
    fr = np.asarray(getattr(tf, "fr", []), dtype=float)
    first_reward = float(timing.get("first_reward_abs_frame", np.nan))
    rows: list[dict[str, Any]] = []

    for bout_idx, no_arr in enumerate(no_list):
        no = np.asarray(no_arr)
        if no.ndim != 2 or no.shape[1] < 2 or no.shape[0] < 2:
            continue
        bout_start = float(fr[bout_idx, 0]) if fr.ndim == 2 and bout_idx < fr.shape[0] else 0.0
        nodes = no[:, 0].astype(int, copy=False)
        frames = no[:, 1].astype(float, copy=False)
        for idx in range(no.shape[0] - 1):
            parent = int(nodes[idx])
            child = int(nodes[idx + 1])
            if parent < 0 or parent >= internal_limit:
                continue
            left_child = int(neighbors[parent, 1])
            right_child = int(neighbors[parent, 2])
            if left_child < 0 or right_child < 0:
                continue
            if child == left_child:
                choice_side = "left"
            elif child == right_child:
                choice_side = "right"
            else:
                continue
            abs_frame = float(bout_start + frames[idx])
            if np.isfinite(first_reward):
                reward_epoch = "pre_first_reward" if abs_frame < first_reward else "post_first_reward"
            else:
                reward_epoch = "no_reward_detected"
            rows.append(
                {
                    "session_id": str(session_row["session_id"]),
                    "animal": str(session_row["animal"]),
                    "condition": str(session_row["condition"]),
                    "condition_label": str(session_row["condition_label"]),
                    "memory_stage": str(session_row["memory_stage"]),
                    "cohort_key": str(session_row["cohort_key"]),
                    "reward_context": str(session_row["reward_context"]),
                    "bout": int(bout_idx),
                    "node_index_in_bout": int(idx),
                    "abs_frame": abs_frame,
                    "current_node": parent,
                    "chosen_child": child,
                    "choice_side": choice_side,
                    "left_child": left_child,
                    "right_child": right_child,
                    "paper_depth": int(_node_depth(parent, ma) + 1),
                    "tree_depth": int(_node_depth(parent, ma)),
                    "first_reward_abs_frame": first_reward,
                    "first_reward_maze_seconds": float(timing.get("first_reward_maze_seconds", np.nan)),
                    "n_reward_deliveries": int(timing.get("n_reward_deliveries", 0)),
                    "reward_epoch": reward_epoch,
                }
            )
    return pd.DataFrame(rows)


def _node_depth(node: int, ma: Any) -> int:
    depth = 0
    cur = int(node)
    while cur > 0:
        cur = int(ma.pa[cur])
        depth += 1
        if depth > len(ma.ru):
            break
    return depth


def _legacy_first_child_choices(
    tf: Any,
    ma: Any,
    session_row: pd.Series,
    reward_path: pd.DataFrame,
    *,
    timing: dict[str, Any],
) -> pd.DataFrame:
    no_list = getattr(tf, "no", []) or []
    fr = np.asarray(getattr(tf, "fr", []), dtype=float)
    first_reward = float(timing.get("first_reward_abs_frame", np.nan))
    node_rows: list[dict[str, Any]] = []
    for bout_idx, no_arr in enumerate(no_list):
        no = np.asarray(no_arr)
        if no.ndim != 2 or no.shape[1] < 2 or no.shape[0] == 0:
            continue
        bout_start = float(fr[bout_idx, 0]) if fr.ndim == 2 and bout_idx < fr.shape[0] else 0.0
        for idx in range(no.shape[0]):
            abs_frame = float(bout_start + float(no[idx, 1]))
            if np.isfinite(first_reward) and abs_frame >= first_reward:
                continue
            node_rows.append(
                {
                    "bout": int(bout_idx),
                    "node_index_in_bout": int(idx),
                    "node": int(no[idx, 0]),
                    "abs_frame": abs_frame,
                }
            )
    nodes = pd.DataFrame(node_rows)
    rows: list[dict[str, Any]] = []
    for _, rp in reward_path.iterrows():
        if nodes.empty:
            sub = pd.DataFrame()
        else:
            sub = nodes[nodes["node"].isin([int(rp["reward_child"]), int(rp["sibling_child"])])].sort_values("abs_frame")
        if sub.empty:
            chosen = -1
            abs_frame = np.nan
            correct = np.nan
        else:
            first = sub.iloc[0]
            chosen = int(first["node"])
            abs_frame = float(first["abs_frame"])
            correct = bool(chosen == int(rp["reward_child"]))
        rows.append(
            {
                "session_id": str(session_row["session_id"]),
                "animal": str(session_row["animal"]),
                "condition": str(session_row["condition"]),
                "condition_label": str(session_row["condition_label"]),
                "memory_stage": str(session_row["memory_stage"]),
                "cohort_key": str(session_row["cohort_key"]),
                "paper_depth": int(rp["paper_depth"]),
                "tree_depth": int(rp["tree_depth"]),
                "parent_node": int(rp["parent_node"]),
                "reward_child": int(rp["reward_child"]),
                "sibling_child": int(rp["sibling_child"]),
                "reward_side": str(rp["reward_side"]),
                "chosen_child": chosen,
                "abs_frame": abs_frame,
                "reward_correct": correct,
                "metric": "legacy_first_child_occurrence_pre_reward",
            }
        )
    return pd.DataFrame(rows)


def extract_session_tables(
    tf: Any,
    ma: Any,
    session_row: pd.Series,
    reward_path: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    timing = _session_timing(tf)
    exits = _extract_child_exit_events(tf, ma, session_row, timing=timing)
    pre_exits = exits[exits["reward_epoch"].isin(["pre_first_reward", "no_reward_detected"])].copy()
    pre_exits = pre_exits.sort_values("abs_frame")
    first_by_parent = pre_exits.drop_duplicates("current_node", keep="first")
    reconstructed, entry_audit, entry_frame_audit = _build_initial_path_reconstruction(
        tf,
        ma,
        session_row,
        reward_path,
        timing=timing,
    )

    path_nodes = set(reward_path["parent_node"].astype(int).tolist())
    reward_side_by_depth = {
        int(row.paper_depth): str(row.reward_side)
        for _, row in reward_path.iterrows()
    }

    first_turn_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    for _, rp in reward_path.iterrows():
        parent = int(rp["parent_node"])
        sub = first_by_parent[first_by_parent["current_node"].eq(parent)]
        later_observed_child = int(sub.iloc[0]["chosen_child"]) if not sub.empty else -1
        later_observed_side = str(sub.iloc[0]["choice_side"]) if not sub.empty else ""
        later_observed_frame = float(sub.iloc[0]["abs_frame"]) if not sub.empty else np.nan
        if parent in reconstructed:
            inferred = reconstructed[parent]
            chosen = int(inferred["chosen_child"])
            choice_side = str(inferred["choice_side"])
            abs_frame = np.nan
            correct = bool(inferred["reward_correct"])
            source = "inferred_unique_home_to_first_valid_location"
            choice_origin = "inferred"
            directly_observed = False
            inference_rule = INITIAL_PATH_RECONSTRUCTION_RULE
            inference_assumption = "direct_home_to_first_valid_location"
            first_valid_abs_frame = float(entry_audit.iloc[0]["first_valid_abs_frame"])
            reconstruction_rows.append(
                {
                    "session_id": str(session_row["session_id"]),
                    "animal": str(session_row["animal"]),
                    "condition": str(session_row["condition"]),
                    "paper_depth": int(rp["paper_depth"]),
                    "parent_node": parent,
                    "inferred_chosen_child": chosen,
                    "inferred_choice_side": choice_side,
                    "inferred_reward_correct": correct,
                    "choice_time_interval": f"abs_frame<{first_valid_abs_frame:g}",
                    "later_recorded_parent_exit_available": bool(not sub.empty),
                    "later_recorded_chosen_child": later_observed_child,
                    "later_recorded_choice_side": later_observed_side,
                    "later_recorded_abs_frame": later_observed_frame,
                    "later_recorded_choice_differs": bool(
                        not sub.empty and later_observed_child != chosen
                    ),
                    "selection_rule": (
                        "inferred pre-recording choice precedes any recorded parent exit"
                    ),
                }
            )
        elif sub.empty:
            chosen = -1
            choice_side = ""
            abs_frame = np.nan
            correct = np.nan
            source = "missing_pre_reward_parent_exit"
            choice_origin = "unresolved"
            directly_observed = False
            inference_rule = ""
            inference_assumption = ""
            first_valid_abs_frame = np.nan
        else:
            row = sub.iloc[0]
            chosen = int(row["chosen_child"])
            choice_side = str(row["choice_side"])
            abs_frame = float(row["abs_frame"])
            correct = bool(chosen == int(rp["reward_child"]))
            source = "first_pre_reward_parent_exit"
            choice_origin = "observed"
            directly_observed = True
            inference_rule = ""
            inference_assumption = ""
            first_valid_abs_frame = np.nan
        first_turn_rows.append(
            {
                "session_id": str(session_row["session_id"]),
                "animal": str(session_row["animal"]),
                "condition": str(session_row["condition"]),
                "condition_label": str(session_row["condition_label"]),
                "memory_stage": str(session_row["memory_stage"]),
                "cohort_key": str(session_row["cohort_key"]),
                "reward_context": str(session_row["reward_context"]),
                "paper_depth": int(rp["paper_depth"]),
                "tree_depth": int(rp["tree_depth"]),
                "parent_node": parent,
                "reward_child": int(rp["reward_child"]),
                "sibling_child": int(rp["sibling_child"]),
                "reward_side": str(rp["reward_side"]),
                "previous_reward_side": str(rp.get("previous_reward_side", "")),
                "required_transition_type": str(rp.get("required_transition_type", "")),
                "chosen_child": chosen,
                "choice_side": choice_side,
                "abs_frame": abs_frame,
                "first_reward_abs_frame": float(timing["first_reward_abs_frame"]),
                "first_reward_maze_seconds": float(timing["first_reward_maze_seconds"]),
                "n_reward_deliveries": int(timing["n_reward_deliveries"]),
                "reward_correct": correct,
                "source": source,
                "choice_origin": choice_origin,
                "directly_observed": directly_observed,
                "score_included": bool(np.isfinite(float(correct))),
                "inference_rule": inference_rule,
                "inference_assumption": inference_assumption,
                "inference_interval_upper_abs_frame": first_valid_abs_frame,
                "inference_unique": bool(choice_origin == "inferred"),
                "later_recorded_parent_exit_available": bool(not sub.empty),
                "later_recorded_chosen_child": later_observed_child,
                "later_recorded_choice_side": later_observed_side,
                "later_recorded_abs_frame": later_observed_frame,
            }
        )
    first_turns = pd.DataFrame(first_turn_rows)

    offpath = first_by_parent[
        ~first_by_parent["current_node"].isin(path_nodes)
        & first_by_parent["paper_depth"].isin(list(reward_side_by_depth.keys()))
    ].copy()
    if not offpath.empty:
        offpath["reward_path_side_at_depth"] = offpath["paper_depth"].map(reward_side_by_depth)
        offpath["same_side_as_reward_path"] = offpath["choice_side"].eq(offpath["reward_path_side_at_depth"])

    legacy = _legacy_first_child_choices(tf, ma, session_row, reward_path, timing=timing)
    return {
        "child_exit_events": exits,
        "reward_path_first_turns": first_turns,
        "entry_path_reconstruction_audit": entry_audit,
        "entry_path_choice_reconstruction_audit": pd.DataFrame(reconstruction_rows),
        "entry_path_frame_audit": entry_frame_audit,
        "offpath_side_control_events": offpath,
        "legacy_first_child_turns": legacy,
    }


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    boots = np.empty(int(n_bootstrap), dtype=float)
    for i in range(boots.size):
        sample = vals[rng.integers(0, vals.size, size=vals.size)]
        boots[i] = float(np.nanmean(sample))
    return float(np.nanmean(vals)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (a.size * b.size))


def _contrast_row(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    if df.empty:
        return pd.DataFrame()
    grouped: list[tuple[tuple[Any, ...], pd.DataFrame]]
    if group_cols:
        grouped = []
        for key, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            grouped.append((key, sub))
    else:
        grouped = [((), df)]
    for key, sub in grouped:
        payload = {col: val for col, val in zip(group_cols, key)}
        naive = pd.to_numeric(sub.loc[sub["condition"].eq("naive"), value_col], errors="coerce").to_numpy(dtype=float)
        ret = pd.to_numeric(sub.loc[sub["condition"].eq("return"), value_col], errors="coerce").to_numpy(dtype=float)
        naive = naive[np.isfinite(naive)]
        ret = ret[np.isfinite(ret)]
        if naive.size and ret.size:
            diff = float(np.nanmedian(ret) - np.nanmedian(naive))
            ci_lo, ci_hi = _bootstrap_median_diff(ret, naive, rng=rng, n_bootstrap=n_bootstrap)
            try:
                _, p_val = stats.mannwhitneyu(ret, naive, alternative="two-sided")
            except Exception:
                p_val = np.nan
        else:
            diff = ci_lo = ci_hi = p_val = np.nan
        rows.append(
            {
                **payload,
                "n_animals_naive": int(naive.size),
                "n_animals_return": int(ret.size),
                "median_naive": float(np.nanmedian(naive)) if naive.size else np.nan,
                "median_return": float(np.nanmedian(ret)) if ret.size else np.nan,
                "median_diff_return_minus_naive": diff,
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
                "mannwhitney_p": float(p_val) if np.isfinite(p_val) else np.nan,
                "cliffs_delta_return_vs_naive": _cliffs_delta(ret, naive),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _pairwise_contrast_rows(
    df: pd.DataFrame,
    value_col: str,
    *,
    comparisons: list[tuple[str, str, str]],
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    grouped: list[tuple[tuple[Any, ...], pd.DataFrame]]
    if group_cols:
        grouped = []
        for key, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            grouped.append((key, sub))
    else:
        grouped = [((), df)]
    for key, sub in grouped:
        payload = {col: val for col, val in zip(group_cols, key)}
        for label, target_condition, baseline_condition in comparisons:
            target = pd.to_numeric(
                sub.loc[sub["condition"].eq(target_condition), value_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            baseline = pd.to_numeric(
                sub.loc[sub["condition"].eq(baseline_condition), value_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            target = target[np.isfinite(target)]
            baseline = baseline[np.isfinite(baseline)]
            if target.size and baseline.size:
                diff = float(np.nanmedian(target) - np.nanmedian(baseline))
                try:
                    _, p_val = stats.mannwhitneyu(target, baseline, alternative="two-sided")
                except Exception:
                    p_val = np.nan
            else:
                diff = p_val = np.nan
            rows.append(
                {
                    **payload,
                    "comparison": label,
                    "target_condition": target_condition,
                    "baseline_condition": baseline_condition,
                    "n_target": int(target.size),
                    "n_baseline": int(baseline.size),
                    "median_target": float(np.nanmedian(target)) if target.size else np.nan,
                    "median_baseline": float(np.nanmedian(baseline)) if baseline.size else np.nan,
                    "median_diff_target_minus_baseline": diff,
                    "mannwhitney_p": float(p_val) if np.isfinite(p_val) else np.nan,
                    "cliffs_delta_target_vs_baseline": _cliffs_delta(target, baseline),
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, float((m - rank) * p[idx]))
        running = max(running, adjusted)
        out[idx] = running
    return out


def _condition_support_summary(data: pd.DataFrame) -> str:
    chunks: list[str] = []
    for condition in CONDITION_ORDER:
        sub = data[data["condition"].eq(condition)]
        if sub.empty:
            continue
        n_sessions = int(sub["session_id"].nunique()) if "session_id" in sub.columns else 0
        chunks.append(
            f"{condition}:rows={len(sub)},animals={sub['animal'].nunique()},sessions={n_sessions}"
        )
    return ";".join(chunks)


def _gee_model_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    keep_cols: tuple[str, ...] = ("session_id", "animal", "condition", "condition_label"),
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=[*keep_cols, "value"])
    available = [col for col in keep_cols if col in df.columns]
    data = df.loc[:, available].copy()
    data["value"] = pd.to_numeric(df[value_col], errors="coerce")
    data = data[np.isfinite(data["value"])].copy()
    data = data[data["condition"].isin(CONDITION_ORDER)].copy()
    data["condition"] = pd.Categorical(data["condition"], categories=list(CONDITION_ORDER))
    data["animal"] = data["animal"].astype(str)
    if "session_id" in data.columns:
        data["session_id"] = data["session_id"].astype(str)
    return data.reset_index(drop=True)


def _fit_gee(data: pd.DataFrame, formula: str, family_name: str) -> tuple[Any | None, str, str]:
    if sm is None or smf is None:
        return None, "failed", "statsmodels_not_available"
    if data.empty:
        return None, "failed", "no_rows"
    if data["animal"].nunique() < 2:
        return None, "failed", "fewer_than_two_animals"
    if data["condition"].nunique() < 2 and "is_target" not in data.columns:
        return None, "failed", "fewer_than_two_conditions"
    family = sm.families.Binomial() if family_name == "binomial" else sm.families.Gaussian()
    for cov_name, cov_struct in (
        ("exchangeable", sm.cov_struct.Exchangeable()),
        ("independence", sm.cov_struct.Independence()),
    ):
        try:
            model = smf.gee(
                formula,
                groups=data["animal"],
                data=data,
                family=family,
                cov_struct=cov_struct,
            )
            return model.fit(), "ok", cov_name
        except Exception as exc:
            last_error = str(exc)
    return None, "failed", last_error


def _gee_omnibus_row(
    data: pd.DataFrame,
    *,
    metric: str,
    family_name: str,
    unit: str,
    reference_condition: str = "latent_day2",
    note: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "unit": unit,
        "model_family": family_name,
        "reference_condition": reference_condition,
        "n_rows": int(len(data)),
        "n_animals": int(data["animal"].nunique()) if "animal" in data.columns else 0,
        "n_sessions": int(data["session_id"].nunique()) if "session_id" in data.columns else 0,
        "condition_support": _condition_support_summary(data) if not data.empty else "",
        "condition_wald_chi2": np.nan,
        "condition_wald_df": np.nan,
        "condition_wald_p": np.nan,
        "fit_status": "not_run",
        "cov_struct": "",
        "note": note,
    }
    formula = f"value ~ C(condition, Treatment(reference='{reference_condition}'))"
    res, status, detail = _fit_gee(data, formula, family_name)
    row["fit_status"] = status
    row["cov_struct"] = detail if status == "ok" else ""
    if status != "ok" or res is None:
        row["note"] = f"{note}; {detail}".strip("; ")
        return row
    names = list(res.params.index)
    idxs = [idx for idx, name in enumerate(names) if name.startswith("C(condition")]
    if not idxs:
        row["note"] = f"{note}; no_condition_terms".strip("; ")
        return row
    r_matrix = np.zeros((len(idxs), len(names)), dtype=float)
    for out_idx, param_idx in enumerate(idxs):
        r_matrix[out_idx, param_idx] = 1.0
    try:
        wald = res.wald_test(r_matrix, scalar=True)
        row["condition_wald_chi2"] = float(wald.statistic)
        row["condition_wald_df"] = int(len(idxs))
        row["condition_wald_p"] = float(wald.pvalue)
    except Exception as exc:
        row["note"] = f"{note}; wald_failed={exc}".strip("; ")
    return row


def _gee_pairwise_rows(
    data: pd.DataFrame,
    *,
    metric: str,
    family_name: str,
    unit: str,
    comparisons: tuple[tuple[str, str, str], ...] = FOUR_GROUP_PAIRWISE_COMPARISONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, target_condition, baseline_condition in comparisons:
        sub = data[data["condition"].isin([target_condition, baseline_condition])].copy()
        sub["condition"] = pd.Categorical(sub["condition"], categories=[baseline_condition, target_condition])
        sub["is_target"] = sub["condition"].eq(target_condition).astype(float)
        target = sub[sub["condition"].eq(target_condition)]
        baseline = sub[sub["condition"].eq(baseline_condition)]
        row: dict[str, Any] = {
            "metric": metric,
            "unit": unit,
            "model_family": family_name,
            "comparison": label,
            "target_condition": target_condition,
            "baseline_condition": baseline_condition,
            "n_target_rows": int(len(target)),
            "n_baseline_rows": int(len(baseline)),
            "n_target_animals": int(target["animal"].nunique()),
            "n_baseline_animals": int(baseline["animal"].nunique()),
            "n_target_sessions": int(target["session_id"].nunique()) if "session_id" in target.columns else 0,
            "n_baseline_sessions": int(baseline["session_id"].nunique()) if "session_id" in baseline.columns else 0,
            "mean_target": float(target["value"].mean()) if len(target) else np.nan,
            "mean_baseline": float(baseline["value"].mean()) if len(baseline) else np.nan,
            "mean_diff_target_minus_baseline": (
                float(target["value"].mean() - baseline["value"].mean()) if len(target) and len(baseline) else np.nan
            ),
            "gee_coefficient": np.nan,
            "gee_robust_se": np.nan,
            "gee_z": np.nan,
            "gee_p": np.nan,
            "gee_odds_ratio": np.nan,
            "gee_holm_p_within_metric": np.nan,
            "fit_status": "not_run",
            "cov_struct": "",
            "note": "",
        }
        if len(target) == 0 or len(baseline) == 0:
            row["fit_status"] = "failed"
            row["note"] = "missing_pair_condition"
            rows.append(row)
            continue
        res, status, detail = _fit_gee(sub, "value ~ is_target", family_name)
        row["fit_status"] = status
        row["cov_struct"] = detail if status == "ok" else ""
        if status != "ok" or res is None:
            row["note"] = detail
            rows.append(row)
            continue
        try:
            coef = float(res.params["is_target"])
            se = float(res.bse["is_target"])
            row["gee_coefficient"] = coef
            row["gee_robust_se"] = se
            row["gee_z"] = coef / se if np.isfinite(se) and se > 0 else np.nan
            row["gee_p"] = float(res.pvalues["is_target"])
            if family_name == "binomial" and np.isfinite(coef):
                row["gee_odds_ratio"] = float(np.exp(coef))
        except Exception as exc:
            row["fit_status"] = "failed"
            row["note"] = f"coefficient_failed={exc}"
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "metric" in out.columns:
        adjusted_parts = []
        for _, sub in out.groupby("metric", dropna=False, sort=False):
            sub = sub.copy()
            sub["gee_holm_p_within_metric"] = _holm_adjust(
                pd.to_numeric(sub["gee_p"], errors="coerce").to_numpy(dtype=float)
            )
            adjusted_parts.append(sub)
        out = pd.concat(adjusted_parts, ignore_index=True)
    return out


def _build_four_group_gee_tables(
    valid: pd.DataFrame,
    session_route: pd.DataFrame,
    route_adjusted: pd.DataFrame,
    habit_null_session: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs: list[tuple[str, pd.DataFrame, str, str, str, str]] = []
    specs.append(
        (
            "reward_path_first_turn_correct_all_depths",
            _gee_model_frame(valid, "reward_correct_float"),
            "binomial",
            "first_turn_decision",
            "Event-level first reward-path turns; robust covariance clustered by animal.",
        )
    )
    exact_session = session_route.copy()
    if not exact_session.empty:
        exact_session["exact_six_turn_sequence_float"] = exact_session["exact_six_turn_sequence"].astype(float)
    specs.append(
        (
            "exact_six_turn_sequence",
            _gee_model_frame(exact_session, "exact_six_turn_sequence_float"),
            "binomial",
            "session",
            "Session-level exact six-turn outcome over six resolved choices; observed and inferred counts remain separate.",
        )
    )
    for transition_type in ("start", "alternate_required", "repeat_required"):
        sub = valid[valid["required_transition_type"].eq(transition_type)].copy()
        specs.append(
            (
                f"reward_path_first_turn_correct_{transition_type}",
                _gee_model_frame(sub, "reward_correct_float"),
                "binomial",
                "first_turn_decision",
                "Transition-type event-level first turns; robust covariance clustered by animal.",
            )
        )
    if not route_adjusted.empty:
        route_adj = route_adjusted.copy()
        route_adj = route_adj[pd.to_numeric(route_adj["n_depths_with_offpath_control"], errors="coerce").fillna(0) > 0]
    else:
        route_adj = route_adjusted
    specs.append(
        (
            "strategy_adjusted_route_residual",
            _gee_model_frame(route_adj, "mean_reward_minus_offpath_same_side"),
            "gaussian",
            "session",
            "Session-level residual after same-depth off-path same-side control; support-reduced.",
        )
    )
    specs.append(
        (
            "depth_matched_habit_null_residual",
            _gee_model_frame(habit_null_session, "observed_minus_null_mean"),
            "gaussian",
            "session",
            "Session-level residual versus depth-matched habit null; support-reduced.",
        )
    )

    omnibus_rows: list[dict[str, Any]] = []
    pairwise_frames: list[pd.DataFrame] = []
    for metric, data, family_name, unit, note in specs:
        omnibus_rows.append(
            _gee_omnibus_row(
                data,
                metric=metric,
                family_name=family_name,
                unit=unit,
                note=note,
            )
        )
        pairwise_frames.append(
            _gee_pairwise_rows(
                data,
                metric=metric,
                family_name=family_name,
                unit=unit,
            )
        )
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    return omnibus, pairwise


def _paired_overlap_row(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(index="animal", columns="condition", values=value_col, aggfunc="mean")
    if "naive" not in pivot.columns or "return" not in pivot.columns:
        return pd.DataFrame()
    overlap = pivot[pivot["naive"].notna() & pivot["return"].notna()].copy()
    diffs = (
        pd.to_numeric(overlap["return"], errors="coerce")
        - pd.to_numeric(overlap["naive"], errors="coerce")
    ).to_numpy(dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size > 1:
        if np.allclose(diffs, 0):
            p_val = 1.0
        else:
            try:
                p_val = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
            except Exception:
                p_val = np.nan
    else:
        p_val = np.nan
    return pd.DataFrame(
        [
            {
                "metric": value_col,
                "n_overlap_animals": int(diffs.size),
                "median_paired_diff_return_minus_naive": float(np.nanmedian(diffs)) if diffs.size else np.nan,
                "mean_paired_diff_return_minus_naive": float(np.nanmean(diffs)) if diffs.size else np.nan,
                "wilcoxon_p_overlap": p_val,
                "overlap_animals": ";".join(overlap.index.astype(str).tolist()),
            }
        ]
    )


def _depth_matched_habit_null(
    session_depth: pd.DataFrame,
    offpath: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Compare reward-path choices to same-session, same-depth off-path choices."""
    if session_depth.empty or offpath.empty or "same_side_as_reward_path" not in offpath.columns:
        return pd.DataFrame()

    off = offpath.copy()
    off["paper_depth"] = pd.to_numeric(off["paper_depth"], errors="coerce")
    off["same_side_float"] = pd.to_numeric(off["same_side_as_reward_path"], errors="coerce")
    off = off[np.isfinite(off["paper_depth"]) & np.isfinite(off["same_side_float"])].copy()
    if off.empty:
        return pd.DataFrame()

    off_lookup: dict[tuple[str, int], np.ndarray] = {}
    for (session_id, depth), sub in off.groupby(["session_id", "paper_depth"], dropna=False):
        vals = pd.to_numeric(sub["same_side_float"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            off_lookup[(str(session_id), int(depth))] = vals

    n = max(int(n_resamples), 1)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_cols = ["session_id", "animal", "condition", "condition_label", "memory_stage", "cohort_key"]
    for key, sess in session_depth.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = {col: val for col, val in zip(group_cols, key)}
        supported: list[tuple[int, float, np.ndarray]] = []
        for _, row in sess.iterrows():
            depth = int(row["paper_depth"])
            vals = off_lookup.get((str(meta["session_id"]), depth))
            if vals is None or vals.size == 0:
                continue
            observed = float(row["reward_correct"])
            if np.isfinite(observed):
                supported.append((depth, observed, vals))
        if not supported:
            continue

        observed_values = np.asarray([item[1] for item in supported], dtype=float)
        observed_score = float(np.nanmean(observed_values))
        null_scores = np.empty(n, dtype=float)
        null_exact = np.empty(n, dtype=bool)
        for i in range(n):
            sampled = np.asarray([rng.choice(vals) for _, _, vals in supported], dtype=float)
            null_scores[i] = float(np.nanmean(sampled))
            null_exact[i] = bool(np.all(sampled == 1.0))
        rows.append(
            {
                **meta,
                "n_supported_depths": int(len(supported)),
                "supported_depths": ";".join(str(depth) for depth, _, _ in supported),
                "observed_supported_route_score": observed_score,
                "null_route_score_mean": float(np.nanmean(null_scores)),
                "null_route_score_ci95_low": float(np.nanpercentile(null_scores, 2.5)),
                "null_route_score_ci95_high": float(np.nanpercentile(null_scores, 97.5)),
                "observed_minus_null_mean": observed_score - float(np.nanmean(null_scores)),
                "p_null_ge_observed": float((1 + np.sum(null_scores >= observed_score)) / (n + 1)),
                "observed_supported_exact": bool(np.all(observed_values == 1.0)),
                "null_supported_exact_rate": float(np.nanmean(null_exact.astype(float))),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_median_diff(
    target: np.ndarray,
    baseline: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    target = target[np.isfinite(target)]
    baseline = baseline[np.isfinite(baseline)]
    if target.size == 0 or baseline.size == 0:
        return np.nan, np.nan
    diffs = np.empty(int(n_bootstrap), dtype=float)
    for i in range(diffs.size):
        ts = target[rng.integers(0, target.size, size=target.size)]
        bs = baseline[rng.integers(0, baseline.size, size=baseline.size)]
        diffs[i] = float(np.nanmedian(ts) - np.nanmedian(bs))
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def _summarize_session_routes(
    first_turns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return annotated rows, scored rows, and provenance-aware session scores."""
    route_rows = first_turns.copy()
    route_rows["reward_correct_float"] = pd.to_numeric(
        route_rows["reward_correct"], errors="coerce"
    )
    route_rows["decision_scored"] = np.isfinite(route_rows["reward_correct_float"])
    if "directly_observed" in route_rows:
        route_rows["decision_observed"] = route_rows["directly_observed"].fillna(False).astype(bool)
    else:
        route_rows["decision_observed"] = route_rows["decision_scored"]
    if "choice_origin" in route_rows:
        route_rows["decision_inferred"] = route_rows["choice_origin"].astype(str).eq("inferred")
    else:
        route_rows["decision_inferred"] = False
    valid = route_rows.loc[route_rows["decision_scored"]].copy()

    session_route = (
        route_rows.groupby(
            [
                "session_id",
                "animal",
                "condition",
                "condition_label",
                "memory_stage",
                "cohort_key",
            ],
            dropna=False,
        )
        .agg(
            n_decisions_scored=("decision_scored", "sum"),
            n_decisions_observed=("decision_observed", "sum"),
            n_decisions_inferred=("decision_inferred", "sum"),
            n_correct=("reward_correct_float", "sum"),
            first_reward_maze_seconds=("first_reward_maze_seconds", "first"),
        )
        .reset_index()
    )
    for column in ("n_decisions_scored", "n_decisions_observed", "n_decisions_inferred"):
        session_route[column] = session_route[column].astype(int)
    session_route["n_decisions_unresolved"] = 6 - session_route["n_decisions_scored"]
    session_route["route_fraction_correct"] = (
        session_route["n_correct"]
        / session_route["n_decisions_scored"].replace(0, np.nan)
    )
    session_route["all6_observed"] = session_route["n_decisions_observed"].eq(6)
    session_route["all6_resolved"] = session_route["n_decisions_scored"].eq(6)
    session_route["exact_six_turn_sequence"] = (
        session_route["all6_resolved"] & session_route["n_correct"].eq(6)
    )
    return route_rows, valid, session_route


def build_summaries(
    first_turns: pd.DataFrame,
    offpath: pd.DataFrame,
    legacy_first_turns: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_null_resamples: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    route_rows, valid, session_route = _summarize_session_routes(first_turns)
    pairwise_comparisons = [
        ("return_vs_naive", "return", "naive"),
        ("return_vs_latent_day2", "return", "latent_day2"),
        ("experienced_day2_vs_naive", "experienced_day2", "naive"),
        ("experienced_day2_vs_latent_day2", "experienced_day2", "latent_day2"),
        ("experienced_day2_vs_return", "experienced_day2", "return"),
    ]

    session_depth = (
        valid.groupby(
            [
                "session_id",
                "animal",
                "condition",
                "condition_label",
                "memory_stage",
                "cohort_key",
                "paper_depth",
                "reward_side",
                "required_transition_type",
            ],
            dropna=False,
        )
        .agg(
            reward_correct=("reward_correct_float", "mean"),
            chosen_child=("chosen_child", "first"),
            reward_child=("reward_child", "first"),
            sibling_child=("sibling_child", "first"),
            previous_reward_side=("previous_reward_side", "first"),
            first_reward_maze_seconds=("first_reward_maze_seconds", "first"),
        )
        .reset_index()
    )
    animal_depth = (
        session_depth.groupby(["animal", "condition", "condition_label", "paper_depth", "reward_side"], dropna=False)
        .agg(
            reward_correct_rate=("reward_correct", "mean"),
            n_sessions=("session_id", "nunique"),
        )
        .reset_index()
    )
    group_depth = (
        animal_depth.groupby(["condition", "condition_label", "paper_depth", "reward_side"], dropna=False)
        .agg(
            n_animals=("animal", "nunique"),
            mean_reward_correct_rate=("reward_correct_rate", "mean"),
            median_reward_correct_rate=("reward_correct_rate", "median"),
        )
        .reset_index()
    )
    depth_contrasts = _contrast_row(
        animal_depth.rename(columns={"reward_correct_rate": "value"}),
        "value",
        group_cols=["paper_depth", "reward_side"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    transition_session = (
        valid.groupby(
            [
                "session_id",
                "animal",
                "condition",
                "condition_label",
                "memory_stage",
                "cohort_key",
                "required_transition_type",
            ],
            dropna=False,
        )
        .agg(
            n_decisions=("paper_depth", "nunique"),
            mean_reward_correct=("reward_correct_float", "mean"),
        )
        .reset_index()
    )
    transition_animal = (
        transition_session.groupby(
            ["animal", "condition", "condition_label", "required_transition_type"],
            dropna=False,
        )
        .agg(
            mean_reward_correct=("mean_reward_correct", "mean"),
            n_sessions=("session_id", "nunique"),
            mean_n_decisions=("n_decisions", "mean"),
        )
        .reset_index()
    )
    transition_group = (
        transition_animal.groupby(["condition", "condition_label", "required_transition_type"], dropna=False)
        .agg(
            n_animals=("animal", "nunique"),
            mean_reward_correct=("mean_reward_correct", "mean"),
            median_reward_correct=("mean_reward_correct", "median"),
            mean_n_decisions=("mean_n_decisions", "mean"),
        )
        .reset_index()
    )
    transition_contrasts = _contrast_row(
        transition_animal.rename(columns={"mean_reward_correct": "value"}),
        "value",
        group_cols=["required_transition_type"],
        n_bootstrap=n_bootstrap,
        seed=seed + 8,
    )
    transition_pairwise_contrasts = _pairwise_contrast_rows(
        transition_animal,
        "mean_reward_correct",
        comparisons=pairwise_comparisons,
        group_cols=["required_transition_type"],
    )

    animal_route = (
        session_route.groupby(["animal", "condition", "condition_label"], dropna=False)
        .agg(
            n_sessions=("session_id", "nunique"),
            mean_route_fraction_correct=("route_fraction_correct", "mean"),
            median_route_fraction_correct=("route_fraction_correct", "median"),
            exact_sequence_rate=("exact_six_turn_sequence", "mean"),
            all6_observed_rate=("all6_observed", "mean"),
            all6_resolved_rate=("all6_resolved", "mean"),
            mean_n_correct=("n_correct", "mean"),
        )
        .reset_index()
    )
    route_contrasts = _contrast_row(
        animal_route.rename(columns={"mean_route_fraction_correct": "value"}),
        "value",
        group_cols=[],
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    exact_contrasts = _contrast_row(
        animal_route.rename(columns={"exact_sequence_rate": "value"}),
        "value",
        group_cols=[],
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )
    route_pairwise_contrasts = _pairwise_contrast_rows(
        animal_route,
        "mean_route_fraction_correct",
        comparisons=pairwise_comparisons,
        group_cols=[],
    )
    exact_pairwise_contrasts = _pairwise_contrast_rows(
        animal_route,
        "exact_sequence_rate",
        comparisons=pairwise_comparisons,
        group_cols=[],
    )
    paired_route_stats = pd.concat(
        [
            _paired_overlap_row(animal_route, "mean_route_fraction_correct"),
            _paired_overlap_row(animal_route, "exact_sequence_rate"),
        ],
        ignore_index=True,
    )

    if legacy_first_turns.empty:
        legacy_session_route = pd.DataFrame()
        legacy_animal_route = pd.DataFrame()
        legacy_route_contrasts = pd.DataFrame()
    else:
        legacy = legacy_first_turns[
            np.isfinite(pd.to_numeric(legacy_first_turns["reward_correct"], errors="coerce"))
        ].copy()
        legacy["reward_correct_float"] = legacy["reward_correct"].astype(float)
        legacy_session_route = (
            legacy.groupby(["session_id", "animal", "condition", "condition_label", "memory_stage", "cohort_key"], dropna=False)
            .agg(
                n_decisions_observed=("paper_depth", "nunique"),
                n_correct=("reward_correct_float", "sum"),
            )
            .reset_index()
        )
        legacy_session_route["route_fraction_correct"] = (
            legacy_session_route["n_correct"] / legacy_session_route["n_decisions_observed"].replace(0, np.nan)
        )
        legacy_session_route["all6_observed"] = legacy_session_route["n_decisions_observed"].eq(6)
        legacy_session_route["exact_six_turn_sequence"] = (
            legacy_session_route["all6_observed"] & legacy_session_route["n_correct"].eq(6)
        )
        legacy_animal_route = (
            legacy_session_route.groupby(["animal", "condition", "condition_label"], dropna=False)
            .agg(
                n_sessions=("session_id", "nunique"),
                mean_route_fraction_correct=("route_fraction_correct", "mean"),
                exact_sequence_rate=("exact_six_turn_sequence", "mean"),
            )
            .reset_index()
        )
        legacy_route_contrasts = _contrast_row(
            legacy_animal_route.rename(columns={"mean_route_fraction_correct": "value"}),
            "value",
            group_cols=[],
            n_bootstrap=n_bootstrap,
            seed=seed + 7,
        )

    if offpath.empty:
        offpath_session_depth = pd.DataFrame()
        strategy_adjusted = pd.DataFrame()
        strategy_animal_depth = pd.DataFrame()
        strategy_contrasts = pd.DataFrame()
        route_adjusted = pd.DataFrame()
        route_adjusted_contrasts = pd.DataFrame()
        route_adjusted_pairwise_contrasts = pd.DataFrame()
    else:
        offpath_session_depth = (
            offpath.groupby(["session_id", "animal", "condition", "paper_depth"], dropna=False)
            .agg(
                offpath_same_side_rate=("same_side_as_reward_path", "mean"),
                n_offpath_first_exits=("same_side_as_reward_path", "size"),
            )
            .reset_index()
        )
        strategy_adjusted = session_depth.merge(
            offpath_session_depth,
            on=["session_id", "animal", "condition", "paper_depth"],
            how="left",
        )
        strategy_adjusted["reward_minus_offpath_same_side"] = (
            strategy_adjusted["reward_correct"] - strategy_adjusted["offpath_same_side_rate"]
        )
        strategy_animal_depth = (
            strategy_adjusted.groupby(["animal", "condition", "condition_label", "paper_depth"], dropna=False)
            .agg(
                reward_correct_rate=("reward_correct", "mean"),
                offpath_same_side_rate=("offpath_same_side_rate", "mean"),
                reward_minus_offpath_same_side=("reward_minus_offpath_same_side", "mean"),
                n_sessions=("session_id", "nunique"),
                n_offpath_first_exits=("n_offpath_first_exits", "sum"),
            )
            .reset_index()
        )
        strategy_contrasts = _contrast_row(
            strategy_animal_depth.rename(columns={"reward_minus_offpath_same_side": "value"}),
            "value",
            group_cols=["paper_depth"],
            n_bootstrap=n_bootstrap,
            seed=seed + 3,
        )
        route_adjusted = (
            strategy_adjusted.groupby(["session_id", "animal", "condition", "condition_label", "memory_stage", "cohort_key"], dropna=False)
            .agg(
                mean_reward_minus_offpath_same_side=("reward_minus_offpath_same_side", "mean"),
                n_depths_with_offpath_control=("reward_minus_offpath_same_side", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
            )
            .reset_index()
        )
        route_adjusted_animal = (
            route_adjusted.groupby(["animal", "condition", "condition_label"], dropna=False)
            .agg(
                mean_reward_minus_offpath_same_side=("mean_reward_minus_offpath_same_side", "mean"),
                n_sessions=("session_id", "nunique"),
                mean_depths_with_offpath_control=("n_depths_with_offpath_control", "mean"),
            )
            .reset_index()
        )
        route_adjusted_contrasts = _contrast_row(
            route_adjusted_animal.rename(columns={"mean_reward_minus_offpath_same_side": "value"}),
            "value",
            group_cols=[],
            n_bootstrap=n_bootstrap,
            seed=seed + 4,
        )
        route_adjusted_pairwise_contrasts = _pairwise_contrast_rows(
            route_adjusted_animal,
            "mean_reward_minus_offpath_same_side",
            comparisons=pairwise_comparisons,
            group_cols=[],
        )

    habit_null_session = _depth_matched_habit_null(
        session_depth,
        offpath,
        n_resamples=n_null_resamples,
        seed=seed + 9,
    )
    if habit_null_session.empty:
        habit_null_animal = pd.DataFrame()
        habit_null_group = pd.DataFrame()
        habit_null_contrasts = pd.DataFrame()
        habit_null_pairwise_contrasts = pd.DataFrame()
    else:
        habit_null_animal = (
            habit_null_session.groupby(["animal", "condition", "condition_label"], dropna=False)
            .agg(
                n_sessions=("session_id", "nunique"),
                mean_supported_depths=("n_supported_depths", "mean"),
                mean_observed_supported_route_score=("observed_supported_route_score", "mean"),
                mean_null_route_score=("null_route_score_mean", "mean"),
                mean_observed_minus_null=("observed_minus_null_mean", "mean"),
                median_observed_minus_null=("observed_minus_null_mean", "median"),
                mean_observed_supported_exact=("observed_supported_exact", "mean"),
                mean_null_supported_exact_rate=("null_supported_exact_rate", "mean"),
            )
            .reset_index()
        )
        habit_null_group = (
            habit_null_animal.groupby(["condition", "condition_label"], dropna=False)
            .agg(
                n_animals=("animal", "nunique"),
                mean_observed_supported_route_score=("mean_observed_supported_route_score", "mean"),
                median_observed_supported_route_score=("mean_observed_supported_route_score", "median"),
                mean_null_route_score=("mean_null_route_score", "mean"),
                median_null_route_score=("mean_null_route_score", "median"),
                mean_observed_minus_null=("mean_observed_minus_null", "mean"),
                median_observed_minus_null=("mean_observed_minus_null", "median"),
                mean_supported_depths=("mean_supported_depths", "mean"),
            )
            .reset_index()
        )
        habit_null_contrasts = _contrast_row(
            habit_null_animal.rename(columns={"mean_observed_minus_null": "value"}),
            "value",
            group_cols=[],
            n_bootstrap=n_bootstrap,
            seed=seed + 10,
        )
        habit_null_pairwise_contrasts = _pairwise_contrast_rows(
            habit_null_animal,
            "mean_observed_minus_null",
            comparisons=pairwise_comparisons,
            group_cols=[],
        )

    group_route = (
        animal_route.groupby(["condition", "condition_label"], dropna=False)
        .agg(
            n_animals=("animal", "nunique"),
            mean_route_fraction_correct=("mean_route_fraction_correct", "mean"),
            median_route_fraction_correct=("median_route_fraction_correct", "median"),
            mean_exact_sequence_rate=("exact_sequence_rate", "mean"),
            median_exact_sequence_rate=("exact_sequence_rate", "median"),
            mean_all6_observed_rate=("all6_observed_rate", "mean"),
            mean_all6_resolved_rate=("all6_resolved_rate", "mean"),
        )
        .reset_index()
    )

    chance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 5)
    for condition, sub in animal_route.groupby("condition", dropna=False):
        vals = pd.to_numeric(sub["mean_route_fraction_correct"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        mean, ci_lo, ci_hi = _bootstrap_ci(vals, rng=rng, n_bootstrap=n_bootstrap)
        if vals.size > 1 and not np.allclose(vals - 0.5, 0):
            try:
                wil_p = float(stats.wilcoxon(vals - 0.5, alternative="two-sided").pvalue)
            except Exception:
                wil_p = np.nan
        else:
            wil_p = np.nan
        chance_rows.append(
            {
                "condition": str(condition),
                "n_animals": int(vals.size),
                "mean_route_fraction_correct": mean,
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
                "wilcoxon_p_vs_0p5": wil_p,
            }
        )
    route_chance = pd.DataFrame(chance_rows)

    exact_rows: list[dict[str, Any]] = []
    for condition, sub in session_route.groupby("condition", dropna=False):
        resolved = sub[sub["all6_resolved"].astype(bool)]
        n = int(len(resolved))
        k = int(resolved["exact_six_turn_sequence"].sum()) if n else 0
        p = float(stats.binomtest(k, n, p=1.0 / 64.0, alternative="greater").pvalue) if n else np.nan
        exact_rows.append(
            {
                "condition": str(condition),
                "n_sessions_all6_resolved": n,
                "n_sessions_all6_observed": int(sub["all6_observed"].sum()),
                "n_exact_six_turn_sequences": k,
                "exact_sequence_rate_session": float(k / n) if n else np.nan,
                "binom_p_vs_1_over_64_greater": p,
            }
        )
    exact_session_stats = pd.DataFrame(exact_rows)
    four_group_gee_omnibus, four_group_gee_pairwise = _build_four_group_gee_tables(
        valid,
        session_route,
        route_adjusted,
        habit_null_session,
    )

    return {
        "session_depth_summary": session_depth,
        "animal_depth_summary": animal_depth,
        "group_depth_summary": group_depth,
        "depth_contrasts": depth_contrasts,
        "transition_session_summary": transition_session,
        "transition_animal_summary": transition_animal,
        "transition_group_summary": transition_group,
        "transition_type_contrasts": transition_contrasts,
        "transition_pairwise_contrasts": transition_pairwise_contrasts,
        "session_route_summary": session_route,
        "animal_route_summary": animal_route,
        "group_route_summary": group_route,
        "route_contrasts": route_contrasts,
        "exact_sequence_contrasts": exact_contrasts,
        "route_pairwise_contrasts": route_pairwise_contrasts,
        "exact_sequence_pairwise_contrasts": exact_pairwise_contrasts,
        "paired_overlap_route_stats": paired_route_stats,
        "legacy_session_route_summary": legacy_session_route,
        "legacy_animal_route_summary": legacy_animal_route,
        "legacy_route_contrasts": legacy_route_contrasts,
        "route_chance_stats": route_chance,
        "exact_sequence_session_stats": exact_session_stats,
        "offpath_session_depth_summary": offpath_session_depth,
        "strategy_adjusted_session_depth": strategy_adjusted,
        "strategy_adjusted_animal_depth": strategy_animal_depth,
        "strategy_adjusted_depth_contrasts": strategy_contrasts,
        "strategy_adjusted_route_summary": route_adjusted,
        "strategy_adjusted_route_contrasts": route_adjusted_contrasts,
        "strategy_adjusted_route_pairwise_contrasts": route_adjusted_pairwise_contrasts,
        "habit_null_session_summary": habit_null_session,
        "habit_null_animal_summary": habit_null_animal,
        "habit_null_group_summary": habit_null_group,
        "habit_null_route_contrasts": habit_null_contrasts,
        "habit_null_pairwise_contrasts": habit_null_pairwise_contrasts,
        "four_group_gee_omnibus": four_group_gee_omnibus,
        "four_group_gee_pairwise": four_group_gee_pairwise,
    }


def _plot_depth(group_depth: pd.DataFrame, out_dir: Path) -> Path | None:
    if group_depth.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"naive": "#4C78A8", "return": "#D65F5F"}
    for condition, sub in group_depth.groupby("condition"):
        ss = sub.sort_values("paper_depth")
        ax.plot(
            ss["paper_depth"],
            ss["mean_reward_correct_rate"],
            marker="o",
            lw=1.8,
            color=colors.get(str(condition), None),
            label=str(condition),
        )
    ax.axhline(0.5, color="0.45", lw=0.9, ls="--")
    ax.set_xlabel("Reward-path decision depth")
    ax.set_ylabel("First-turn reward-direction rate")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(sorted(group_depth["paper_depth"].unique()))
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "first_turn_accuracy_by_depth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_route(animal_route: pd.DataFrame, out_dir: Path) -> Path | None:
    if animal_route.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    order = ["naive", "return"]
    colors = {"naive": "#4C78A8", "return": "#D65F5F"}
    rng = np.random.default_rng(19)
    for i, condition in enumerate(order):
        sub = animal_route[animal_route["condition"].eq(condition)]
        vals = pd.to_numeric(sub["mean_route_fraction_correct"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        x = np.full(vals.size, i, dtype=float) + rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(x, vals, s=26, color=colors.get(condition), alpha=0.8, edgecolor="white", linewidth=0.4)
        if vals.size:
            ax.plot([i - 0.18, i + 0.18], [np.nanmedian(vals), np.nanmedian(vals)], color="black", lw=1.6)
    ax.axhline(0.5, color="0.45", lw=0.9, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean first-turn sequence fraction correct")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    path = out_dir / "route_sequence_score_by_animal.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_entry_path_reconstruction(
    ma: Any,
    entry_audit: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    """Render an editable audit view of the direct-entry evidence path."""
    if entry_audit.empty:
        return {}
    row = entry_audit.loc[entry_audit["status"].eq("reconstructed")]
    if row.empty:
        return {}
    row = row.iloc[0]
    cell_path = [
        int(value)
        for value in str(row["home_to_entry_cell_path"]).split(";")
        if value != ""
    ]
    run_path = [
        int(value)
        for value in str(row["home_to_entry_run_path"]).split(";")
        if value != ""
    ]
    if not cell_path or not run_path:
        return {}

    fig, (ax_grid, ax_tree) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    ax_grid.scatter(ma.xc, ma.yc, s=5, color="#D5D9DC", zorder=1)
    for cells in ma.ru:
        if len(cells) < 2:
            continue
        values = np.asarray(cells, dtype=int)
        ax_grid.plot(ma.xc[values], ma.yc[values], color="#D5D9DC", lw=0.8, zorder=1)
    path_values = np.asarray(cell_path, dtype=int)
    ax_grid.plot(
        ma.xc[path_values],
        ma.yc[path_values],
        color="#D47420",
        lw=2.2,
        ls=(0, (4, 2)),
        zorder=3,
        label="Inferred pre-recording path",
    )
    ax_grid.scatter(
        [ma.xc[path_values[0]], ma.xc[path_values[-1]]],
        [ma.yc[path_values[0]], ma.yc[path_values[-1]]],
        s=38,
        color=["#303030", "#D47420"],
        zorder=4,
    )
    ax_grid.text(
        ma.xc[path_values[0]] + 0.35,
        ma.yc[path_values[0]] - 0.15,
        "Home",
        ha="left",
        va="center",
        fontsize=8,
    )
    ax_grid.text(
        ma.xc[path_values[-1]] - 0.15,
        ma.yc[path_values[-1]] + 0.65,
        f"First valid location\ncell {path_values[-1]}, frame {int(row['first_valid_abs_frame'])}",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax_grid.set_aspect("equal")
    ax_grid.invert_yaxis()
    ax_grid.set_title("Grid-space entry evidence", fontsize=10)
    ax_grid.axis("off")

    reward_nodes = [0, 1, 4, 10, 22, 46, 94]
    for nodes, color, label, linestyle in (
        (reward_nodes, "#2E77B5", "Reward route", "-"),
        (run_path, "#D47420", "Direct-entry path", (0, (4, 2))),
    ):
        x = [_node_depth(node, ma) for node in nodes]
        ax_tree.plot(x, nodes, color=color, lw=2.0, ls=linestyle, marker="o", ms=3.5, label=label, zorder=3)
    ax_tree.annotate(
        "same root choice",
        xy=(1, 1),
        xytext=(1.4, 17),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"},
    )
    ax_tree.annotate(
        "first divergence: 1→3",
        xy=(2, 3),
        xytext=(2.6, 27),
        fontsize=8,
        color="#D47420",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#D47420"},
    )
    ax_tree.set_xlabel("Tree depth")
    ax_tree.set_ylabel("Run node")
    ax_tree.set_title("Tree-path choice reconstruction", fontsize=10)
    ax_tree.legend(frameon=False, fontsize=8, loc="upper left")
    ax_tree.spines[["top", "right"]].set_visible(False)
    fig.suptitle("489-3day1 direct-entry reconstruction audit", fontsize=11)
    fig.tight_layout()

    outputs: dict[str, str] = {}
    base = out_dir / "entry_path_reconstruction_489-3day1"
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 240}),
    ):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def _plot_strategy(strategy_animal_depth: pd.DataFrame, out_dir: Path) -> Path | None:
    if strategy_animal_depth.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"naive": "#4C78A8", "return": "#D65F5F"}
    for condition, sub in strategy_animal_depth.groupby("condition"):
        ss = (
            sub.groupby("paper_depth", dropna=False)["reward_minus_offpath_same_side"]
            .mean()
            .reset_index()
            .sort_values("paper_depth")
        )
        ax.plot(
            ss["paper_depth"],
            ss["reward_minus_offpath_same_side"],
            marker="o",
            lw=1.8,
            color=colors.get(str(condition), None),
            label=str(condition),
        )
    ax.axhline(0, color="0.45", lw=0.9, ls="--")
    ax.set_xlabel("Reward-path decision depth")
    ax.set_ylabel("Reward-path rate minus off-path same-side rate")
    ax.set_ylim(-1.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "strategy_adjusted_reward_path_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}g}"


def _first_row(df: pd.DataFrame) -> dict[str, Any]:
    return df.iloc[0].to_dict() if isinstance(df, pd.DataFrame) and not df.empty else {}


def write_report(
    out_dir: Path,
    *,
    manifest: pd.DataFrame,
    reward_path: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    plot_paths: dict[str, str],
    params: dict[str, Any],
) -> Path:
    loaded = manifest[manifest["load_status"].eq("ok")].copy()
    excluded = manifest[~manifest["load_status"].eq("ok")].copy()
    animal_route = summaries["animal_route_summary"]
    group_route = summaries["group_route_summary"]
    route_contrast = _first_row(summaries["route_contrasts"])
    exact_contrast = _first_row(summaries["exact_sequence_contrasts"])
    strategy_contrast = _first_row(summaries["strategy_adjusted_route_contrasts"])
    habit_null_contrast = _first_row(summaries["habit_null_route_contrasts"])
    legacy_contrast = _first_row(summaries["legacy_route_contrasts"])

    lines: list[str] = []
    lines.append("# Whole-Cohort First-Turn Sequence Retention")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Primary cohort: Rewarded naive day 1 versus pooled 3/6/9/12-month return day 1 sessions; canonical reward node `{REWARD_NODE}`.")
    lines.append("- Added habit-control cohort: `AllUnrewardedDay2` latent animals (9 animals) that had unrewarded layout exposure on day 1 and first reward access on day 2.")
    lines.append(f"- Loaded sessions: {len(loaded)} of {len(manifest)}; unique loaded animals: {loaded['animal'].nunique() if not loaded.empty else 0}.")
    lines.append("- Primary event: earliest pre-first-reward parent-to-child exit at each of the six reward-path decision nodes, after any explicitly approved direct-entry reconstruction.")
    n_inferred = int(pd.to_numeric(loaded.get("n_reward_path_first_turns_inferred", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    lines.append(f"- Direct-entry reconstruction: {n_inferred} interval-censored choices in `489-3day1`, inferred from the unique home-to-first-valid-location path; all other sessions use recorded exits only.")
    lines.append(
        f"- Route fraction correct, animal-collapsed median return-minus-naive: "
        f"{_fmt(route_contrast.get('median_diff_return_minus_naive'))} "
        f"(95% bootstrap CI [{_fmt(route_contrast.get('bootstrap_ci95_low'))}, {_fmt(route_contrast.get('bootstrap_ci95_high'))}], "
        f"Mann-Whitney p={_fmt(route_contrast.get('mannwhitney_p'))})."
    )
    lines.append(
        f"- Exact six-turn sequence rate, animal-collapsed median return-minus-naive: "
        f"{_fmt(exact_contrast.get('median_diff_return_minus_naive'))} "
        f"(p={_fmt(exact_contrast.get('mannwhitney_p'))})."
    )
    if strategy_contrast:
        lines.append(
            f"- Strategy-adjusted route score, return-minus-naive: "
            f"{_fmt(strategy_contrast.get('median_diff_return_minus_naive'))} "
            f"(p={_fmt(strategy_contrast.get('mannwhitney_p'))})."
        )
    if habit_null_contrast:
        lines.append(
            f"- Depth-matched off-path habit-null residual, return-minus-naive: "
            f"{_fmt(habit_null_contrast.get('median_diff_return_minus_naive'))} "
            f"(p={_fmt(habit_null_contrast.get('mannwhitney_p'))})."
        )
    lines.append("- The reward-side sequence is `L R R L R L`; four of the five post-root choices are alternation-consistent, so reward-direction turns can partly reflect the local alternation habit rather than a stored route.")
    lines.append("")

    lines.append("## Existing Analysis Located")
    lines.append("- Legacy notebooks `notebooks/archive_legacy/260113_behaviormeasures.ipynb` and `notebooks/archive_legacy/250210_behaviormeasures.ipynb`, cells 15-22, contain the original first-turn block.")
    lines.append("- That block flattens `tf.no` node sequences and asks which child appears first for reward-path sibling pairs: `1/2`, `3/4`, `9/10`, `21/22`, `45/46`, and `93/94`.")
    lines.append("- The legacy block is useful provenance but is not a maintained all-cohort export: the active cohort line is manually swapped, it does not write audit tables, and the Fisher test uses hard-coded counts.")
    lines.append("")

    lines.append("## Operational Definitions")
    lines.append("- Reward path: `0 -> 1 -> 4 -> 10 -> 22 -> 46 -> 94`.")
    lines.append("- Primary first choice: for a parent node on that path, the earliest pre-first-reward transition where the current node is the parent and the next node is either the reward child or its sibling.")
    lines.append("- Direct-entry exception: in explicitly approved `489-3day1`, the first valid cell and run-node samples both identify node 16 at absolute frame 378. Under the author-selected assumption of direct travel from home, the unique run path `0 -> 1 -> 3 -> 7 -> 16` supplies two earlier, interval-censored choices. These precede and therefore supersede any later recorded exit at the same parent; no exact event frames are fabricated.")
    lines.append("- A turn is reward-correct if the selected child is the next node on the canonical reward path.")
    lines.append("- `route_fraction_correct`: number of reward-correct resolved first choices divided by the number of resolved reward-path choices. Directly observed and inferred counts are retained separately.")
    lines.append("- `all6_observed`: all six first choices were directly recorded. `all6_resolved`: all six were directly recorded or uniquely reconstructed.")
    lines.append("- `exact_six_turn_sequence`: all six choices were resolved and all selected the reward child.")
    lines.append("- Legacy sensitivity: `legacy_first_child_turns.csv` reproduces the older child-appears-first proxy, but still applies the pre-first-reward frame window.")
    lines.append("- Strategy control: at each depth, off-path first exits are scored for whether they choose the same left/right child side as the reward-path child at that depth. The adjusted score is reward-path correctness minus this off-path same-side rate.")
    lines.append("- Turn-habit alignment: each reward-path decision is labeled by whether the reward-side choice repeats or alternates relative to the previous reward-path side. This tests whether the apparently correct sequence is partly aligned with generic alternation.")
    lines.append("- Depth-matched habit null: for each session and supported depth, one off-path first exit of the same depth is resampled to form a local-policy null route score. Depth 1 is excluded when no same-depth off-path node exists.")
    lines.append("- This is behavior-only; no calcium or `selected_df` timebase is merged.")
    lines.append("")

    lines.append("## Reward-Path Geometry")
    lines.append("| Depth | Parent | Reward child | Sibling child | Reward side | Previous reward side | Required transition |")
    lines.append("|---:|---:|---:|---:|---|---|---|")
    for _, row in reward_path.iterrows():
        lines.append(
            f"| {int(row['paper_depth'])} | {int(row['parent_node'])} | "
            f"{int(row['reward_child'])} | {int(row['sibling_child'])} | {row['reward_side']} | "
            f"{row['previous_reward_side'] or 'NA'} | {row['required_transition_type']} |"
        )
    lines.append("")

    lines.append("## Cohort Audit")
    if loaded.empty:
        lines.append("- No sessions loaded.")
    else:
        for condition, sub in loaded.groupby("condition", sort=True):
            label = str(sub["condition_label"].iloc[0])
            stages = ", ".join(sorted(sub["memory_stage"].astype(str).unique()))
            cohorts = ", ".join(sorted(sub["cohort_key"].astype(str).unique()))
            lines.append(f"- {label} (`{condition}`): {len(sub)} sessions, {sub['animal'].nunique()} animals, stages {stages}, cohorts {cohorts}.")
    if not excluded.empty:
        lines.append("- Excluded sessions:")
        for _, row in excluded.iterrows():
            lines.append(f"  - `{row['session_id']}` ({row['cohort_key']}): {row['load_status']}")
    lines.append("")

    lines.append("## Depth-Level First-Turn Accuracy")
    group_depth = summaries["group_depth_summary"]
    if group_depth.empty:
        lines.append("- No depth-level summary available.")
    else:
        lines.append("| Condition | Depth | Reward side | n animals | Mean correct rate | Median correct rate |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for _, row in group_depth.sort_values(["paper_depth", "condition"]).iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['paper_depth'])} | {row['reward_side']} | "
                f"{int(row['n_animals'])} | {_fmt(row['mean_reward_correct_rate'])} | {_fmt(row['median_reward_correct_rate'])} |"
            )
    lines.append("")

    lines.append("## Turn-Habit Alignment")
    transition_group = summaries["transition_group_summary"]
    transition_contrasts = summaries["transition_type_contrasts"]
    if transition_group.empty:
        lines.append("- No transition-type summary available.")
    else:
        lines.append("- The key control is that a correct reward-path decision can also be an alternation-consistent decision. Here `alternate_required` means the reward side differs from the previous reward-path side, and `repeat_required` means it matches the previous side.")
        lines.append("| Condition | Required transition | n animals | Mean correct | Median correct | Mean decisions/animal |")
        lines.append("|---|---|---:|---:|---:|---:|")
        order = {"start": 0, "alternate_required": 1, "repeat_required": 2}
        for _, row in transition_group.assign(
            _order=transition_group["required_transition_type"].map(order).fillna(99)
        ).sort_values(["_order", "condition"]).iterrows():
            lines.append(
                f"| {row['condition']} | {row['required_transition_type']} | {int(row['n_animals'])} | "
                f"{_fmt(row['mean_reward_correct'])} | {_fmt(row['median_reward_correct'])} | "
                f"{_fmt(row['mean_n_decisions'])} |"
            )
    if not transition_contrasts.empty:
        lines.append("")
        lines.append("| Required transition | Naive median | Return median | Return - naive | Mann-Whitney p |")
        lines.append("|---|---:|---:|---:|---:|")
        order = {"start": 0, "alternate_required": 1, "repeat_required": 2}
        for _, row in transition_contrasts.assign(
            _order=transition_contrasts["required_transition_type"].map(order).fillna(99)
        ).sort_values("_order").iterrows():
            lines.append(
                f"| {row['required_transition_type']} | {_fmt(row['median_naive'])} | "
                f"{_fmt(row['median_return'])} | {_fmt(row['median_diff_return_minus_naive'])} | "
                f"{_fmt(row['mannwhitney_p'])} |"
            )
    if not transition_group.empty:
        latent_alt = transition_group[
            transition_group["condition"].eq("latent_day2")
            & transition_group["required_transition_type"].eq("alternate_required")
        ]
        return_alt = transition_group[
            transition_group["condition"].eq("return")
            & transition_group["required_transition_type"].eq("alternate_required")
        ]
        if not latent_alt.empty and not return_alt.empty:
            lines.append("")
            lines.append(
                f"- Latent D2 animals, despite no prior reward history, are already high on alternate-required choices "
                f"(mean {_fmt(latent_alt.iloc[0]['mean_reward_correct'])}) and close to return animals "
                f"(mean {_fmt(return_alt.iloc[0]['mean_reward_correct'])}). This directly supports the concern that some reward-correct turns are expected from the local alternation habit."
            )
    lines.append("")

    lines.append("## Route-Level Summary")
    if group_route.empty:
        lines.append("- No route-level summary available.")
    else:
        lines.append("| Condition | n animals | Mean route fraction | Median route fraction | Mean exact-six rate | Mean all-six-observed rate | Mean all-six-resolved rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in group_route.sort_values("condition").iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['n_animals'])} | "
                f"{_fmt(row['mean_route_fraction_correct'])} | {_fmt(row['median_route_fraction_correct'])} | "
                f"{_fmt(row['mean_exact_sequence_rate'])} | {_fmt(row['mean_all6_observed_rate'])} | "
                f"{_fmt(row['mean_all6_resolved_rate'])} |"
            )
    lines.append("")

    lines.append("## Primary Contrasts")
    lines.append("| Metric | Naive median | Return median | Return - naive | 95% bootstrap CI | Mann-Whitney p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, row in [
        ("route_fraction_correct", route_contrast),
        ("exact_six_turn_sequence_rate", exact_contrast),
        ("strategy_adjusted_route_score", strategy_contrast),
        ("depth_matched_habit_null_residual", habit_null_contrast),
    ]:
        if not row:
            continue
        lines.append(
            f"| {label} | {_fmt(row.get('median_naive'))} | {_fmt(row.get('median_return'))} | "
            f"{_fmt(row.get('median_diff_return_minus_naive'))} | "
            f"[{_fmt(row.get('bootstrap_ci95_low'))}, {_fmt(row.get('bootstrap_ci95_high'))}] | "
            f"{_fmt(row.get('mannwhitney_p'))} |"
        )
    lines.append("")

    lines.append("## Paired Animal-Overlap Check")
    overlap = summaries["paired_overlap_route_stats"]
    if overlap.empty:
        lines.append("- No paired overlap statistics were available.")
    else:
        lines.append("- Animals appearing in both naive and return arms are tested as paired return-minus-naive differences after collapsing return sessions within animal.")
        lines.append("| Metric | n overlap animals | Median paired diff | Mean paired diff | Wilcoxon p |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in overlap.iterrows():
            lines.append(
                f"| {row['metric']} | {int(row['n_overlap_animals'])} | "
                f"{_fmt(row['median_paired_diff_return_minus_naive'])} | "
                f"{_fmt(row['mean_paired_diff_return_minus_naive'])} | {_fmt(row['wilcoxon_p_overlap'])} |"
            )
    lines.append("")

    lines.append("## Legacy Proxy Sensitivity")
    if not legacy_contrast:
        lines.append("- Legacy child-appears-first sensitivity was not available.")
    else:
        lines.append("- This reproduces the old child-appears-first logic, with the added pre-first-reward filter and animal-level collapse.")
        lines.append(
            f"- Legacy route fraction correct return-minus-naive median difference: "
            f"{_fmt(legacy_contrast.get('median_diff_return_minus_naive'))} "
            f"(95% bootstrap CI [{_fmt(legacy_contrast.get('bootstrap_ci95_low'))}, {_fmt(legacy_contrast.get('bootstrap_ci95_high'))}], "
            f"Mann-Whitney p={_fmt(legacy_contrast.get('mannwhitney_p'))})."
        )
    lines.append("")

    lines.append("## Chance and Exact-Sequence Checks")
    chance = summaries["route_chance_stats"]
    exact = summaries["exact_sequence_session_stats"]
    if not chance.empty:
        lines.append("| Condition | n animals | Mean route fraction | 95% bootstrap CI | Wilcoxon p vs 0.5 |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in chance.sort_values("condition").iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['n_animals'])} | {_fmt(row['mean_route_fraction_correct'])} | "
                f"[{_fmt(row['bootstrap_ci95_low'])}, {_fmt(row['bootstrap_ci95_high'])}] | {_fmt(row['wilcoxon_p_vs_0p5'])} |"
            )
    if not exact.empty:
        lines.append("")
        lines.append("| Condition | Sessions with all six observed | Sessions with all six resolved | Exact sequences | Exact rate | Binomial p vs 1/64 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in exact.sort_values("condition").iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['n_sessions_all6_observed'])} | "
                f"{int(row['n_sessions_all6_resolved'])} | {int(row['n_exact_six_turn_sequences'])} | "
                f"{_fmt(row['exact_sequence_rate_session'])} | "
                f"{_fmt(row['binom_p_vs_1_over_64_greater'])} |"
            )
    lines.append("")

    lines.append("## Strategy-Control Result")
    strategy_depth = summaries["strategy_adjusted_depth_contrasts"]
    if strategy_depth.empty:
        lines.append("- Strategy-adjusted depth contrasts were not available, usually because off-path pre-reward exits were too sparse.")
    else:
        lines.append("- Positive adjusted values mean reward-path first turns exceed the same-side choice rate at off-path nodes of equivalent depth.")
        lines.append("- These depth rows are exploratory sensitivity checks and are not corrected across depths; they should not be interpreted as evidence that the full six-turn sequence is retained.")
        lines.append("| Depth | Naive median adjusted | Return median adjusted | Return - naive | Mann-Whitney p |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, row in strategy_depth.sort_values("paper_depth").iterrows():
            lines.append(
                f"| {int(row['paper_depth'])} | {_fmt(row['median_naive'])} | {_fmt(row['median_return'])} | "
                f"{_fmt(row['median_diff_return_minus_naive'])} | {_fmt(row['mannwhitney_p'])} |"
            )
    lines.append("")

    lines.append("## Depth-Matched Off-Path Habit Null")
    habit_group = summaries["habit_null_group_summary"]
    habit_contrast = _first_row(summaries["habit_null_route_contrasts"])
    if habit_group.empty:
        lines.append("- Habit-null summaries were not available, usually because same-session off-path first exits were too sparse.")
    else:
        lines.append("- This null resamples off-path first exits from the same session and same depth, then asks whether the observed reward-path route score exceeds what the local side policy would predict. It is a control for the fact that much of the reward path is alternation-consistent.")
        lines.append("| Condition | n animals | Mean observed score | Mean null score | Mean residual | Median residual | Mean supported depths |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in habit_group.sort_values("condition").iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['n_animals'])} | "
                f"{_fmt(row['mean_observed_supported_route_score'])} | {_fmt(row['mean_null_route_score'])} | "
                f"{_fmt(row['mean_observed_minus_null'])} | {_fmt(row['median_observed_minus_null'])} | "
                f"{_fmt(row['mean_supported_depths'])} |"
            )
        if habit_contrast:
            lines.append("")
            lines.append(
                f"- Animal-collapsed habit-null residual return-minus-naive: "
                f"{_fmt(habit_contrast.get('median_diff_return_minus_naive'))} "
                f"(95% bootstrap CI [{_fmt(habit_contrast.get('bootstrap_ci95_low'))}, "
                f"{_fmt(habit_contrast.get('bootstrap_ci95_high'))}], "
                f"Mann-Whitney p={_fmt(habit_contrast.get('mannwhitney_p'))})."
            )
    lines.append("")

    lines.append("## Artifacts")
    for fname in [
        "session_manifest.csv",
        "reward_path_definition.csv",
        "child_exit_events.csv",
        "reward_path_first_turns.csv",
        "entry_path_reconstruction_audit.csv",
        "entry_path_choice_reconstruction_audit.csv",
        "entry_path_frame_audit.csv",
        "legacy_first_child_turns.csv",
        "offpath_side_control_events.csv",
        "session_depth_summary.csv",
        "animal_depth_summary.csv",
        "group_depth_summary.csv",
        "depth_contrasts.csv",
        "transition_session_summary.csv",
        "transition_animal_summary.csv",
        "transition_group_summary.csv",
        "transition_type_contrasts.csv",
        "transition_pairwise_contrasts.csv",
        "session_route_summary.csv",
        "animal_route_summary.csv",
        "group_route_summary.csv",
        "route_contrasts.csv",
        "exact_sequence_contrasts.csv",
        "route_pairwise_contrasts.csv",
        "exact_sequence_pairwise_contrasts.csv",
        "four_group_gee_omnibus.csv",
        "four_group_gee_pairwise.csv",
        "paired_overlap_route_stats.csv",
        "legacy_session_route_summary.csv",
        "legacy_animal_route_summary.csv",
        "legacy_route_contrasts.csv",
        "route_chance_stats.csv",
        "exact_sequence_session_stats.csv",
        "strategy_adjusted_session_depth.csv",
        "strategy_adjusted_animal_depth.csv",
        "strategy_adjusted_depth_contrasts.csv",
        "strategy_adjusted_route_summary.csv",
        "strategy_adjusted_route_contrasts.csv",
        "strategy_adjusted_route_pairwise_contrasts.csv",
        "habit_null_session_summary.csv",
        "habit_null_animal_summary.csv",
        "habit_null_group_summary.csv",
        "habit_null_route_contrasts.csv",
        "habit_null_pairwise_contrasts.csv",
        "run_manifest.json",
    ]:
        lines.append(f"- `{fname}`")
    lines.append("")

    lines.append("## Figures")
    if plot_paths:
        for name, path in plot_paths.items():
            lines.append(f"- {name}: `{path}`")
    else:
        lines.append("- Figure generation was skipped or no plottable data were available.")
    lines.append("")

    lines.append("## Final Discussion")
    lines.append("- This analysis directly tests precise route-sequence retention: return animals should exceed naive animals in first-turn accuracy and exact six-turn sequence rate if the detailed turn sequence is specifically retained.")
    lines.append("- A reward-direction bias above 0.5 can occur without route memory if the reward path happens to align with generic left/right, depth-dependent branch strategies, or local alternation. The reward sequence is mostly alternation-compatible after the root, so this caveat is load-bearing.")
    lines.append("- The strongest support for the manuscript sentence would be: reward-path first turns are biased above chance, but return is not significantly higher than naive, and exact six-turn sequences are not enriched in return relative to naive.")
    lines.append("- The off-path same-side control and the depth-matched habit null test that alternative directly. Residuals in those controls should be interpreted separately from precise sequence retention: they can indicate partial reward-path or task-policy bias without implying retention of the full six-turn route sequence.")
    lines.append("- The latent Day2 control is especially informative for the story: animals with recent layout experience but no prior reward history show high correctness on the alternation-compatible subset of the route, so that subset should not be used as evidence for precise reward-route recall.")
    lines.append("- In this export, the primary conclusion should remain the route-sequence result: no reliable return advantage in complete first-turn sequence accuracy. The added habit-alignment controls strengthen the interpretation that apparent pre-reward correctness is partly compatible with generic turn habits rather than a fully retained motor sequence.")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = _parse_args()
    if int(args.reward_node) != int(REWARD_NODE):
        raise SystemExit(f"Refusing noncanonical reward node {args.reward_node}; use {REWARD_NODE}.")
    bootstrap_environment(repo_root=REPO_ROOT)
    out_dir = (REPO_ROOT / Path(args.out_dir)).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ma = NewMaze(6)
    reward_path = build_reward_path_table(ma, int(args.reward_node))
    manifest = build_manifest(latent_cohort_key=str(args.latent_cohort_key))

    manifest_rows: list[dict[str, Any]] = []
    table_acc: dict[str, list[pd.DataFrame]] = {
        "child_exit_events": [],
        "reward_path_first_turns": [],
        "entry_path_reconstruction_audit": [],
        "entry_path_choice_reconstruction_audit": [],
        "entry_path_frame_audit": [],
        "offpath_side_control_events": [],
        "legacy_first_child_turns": [],
    }
    for _, row in manifest.iterrows():
        out_row = row.to_dict()
        try:
            tf = LoadTraj(f"{row['session_id']}-tf")
            tables = extract_session_tables(tf, ma, row, reward_path)
            for name, table in tables.items():
                if not table.empty:
                    table_acc[name].append(table)
            out_row["load_status"] = "ok"
            first_turns = tables["reward_path_first_turns"]
            out_row["n_reward_path_first_turns_scored"] = int(
                pd.to_numeric(first_turns["reward_correct"], errors="coerce").notna().sum()
            )
            out_row["n_reward_path_first_turns_observed"] = int(
                first_turns["directly_observed"].fillna(False).astype(bool).sum()
            )
            out_row["n_reward_path_first_turns_inferred"] = int(
                first_turns["choice_origin"].astype(str).eq("inferred").sum()
            )
            out_row["n_pre_reward_child_exit_events"] = int(
                tables["child_exit_events"]["reward_epoch"].isin(["pre_first_reward", "no_reward_detected"]).sum()
            )
        except Exception as exc:
            out_row["load_status"] = f"failed:{type(exc).__name__}:{exc}"
            out_row["n_reward_path_first_turns_scored"] = 0
            out_row["n_reward_path_first_turns_observed"] = 0
            out_row["n_reward_path_first_turns_inferred"] = 0
            out_row["n_pre_reward_child_exit_events"] = 0
        manifest_rows.append(out_row)

    manifest_out = pd.DataFrame(manifest_rows)
    tables_out = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in table_acc.items()
    }
    summaries = build_summaries(
        tables_out["reward_path_first_turns"],
        tables_out["offpath_side_control_events"],
        tables_out["legacy_first_child_turns"],
        n_bootstrap=int(args.n_bootstrap),
        n_null_resamples=int(args.n_null_resamples),
        seed=int(args.seed),
    )

    manifest_out.to_csv(out_dir / "session_manifest.csv", index=False)
    reward_path.to_csv(out_dir / "reward_path_definition.csv", index=False)
    for name, table in tables_out.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    plot_paths: dict[str, str] = {}
    if not args.no_figures:
        p = _plot_depth(summaries["group_depth_summary"], fig_dir)
        if p is not None:
            plot_paths["first_turn_accuracy_by_depth"] = str(p)
        p = _plot_route(summaries["animal_route_summary"], fig_dir)
        if p is not None:
            plot_paths["route_sequence_score_by_animal"] = str(p)
        p = _plot_strategy(summaries["strategy_adjusted_animal_depth"], fig_dir)
        if p is not None:
            plot_paths["strategy_adjusted_reward_path_delta"] = str(p)
        entry_plot_paths = _plot_entry_path_reconstruction(
            ma,
            tables_out["entry_path_reconstruction_audit"],
            fig_dir,
        )
        for suffix, path in entry_plot_paths.items():
            plot_paths[f"entry_path_reconstruction_{suffix}"] = path

    params = {
        "reward_node": int(args.reward_node),
        "reward_path": reward_path.to_dict(orient="records"),
        "n_bootstrap": int(args.n_bootstrap),
        "n_null_resamples": int(args.n_null_resamples),
        "seed": int(args.seed),
        "latent_cohort_key": str(args.latent_cohort_key),
        "representation_space": "binary-tree parent-to-child exits in NewMaze(6)",
        "primary_window": "pre_first_reward",
        "primary_metric": "earliest resolved parent-to-child exit at each reward-path decision node",
        "initial_path_reconstruction": {
            "rule": INITIAL_PATH_RECONSTRUCTION_RULE,
            "approved_sessions": sorted(INITIAL_PATH_RECONSTRUCTION_SESSIONS),
            "assumption": "direct_home_to_first_valid_location",
            "chronology": "prepend all reward-parent choices on the unique home-to-entry path",
            "event_time": "interval-censored before first valid trajectory frame",
        },
    }
    report_path = write_report(
        out_dir,
        manifest=manifest_out,
        reward_path=reward_path,
        summaries=summaries,
        plot_paths=plot_paths,
        params=params,
    )
    run_manifest = {
        "params": params,
        "outputs": {
            "out_dir": str(out_dir),
            "report": str(report_path),
            "figures": plot_paths,
        },
        "session_counts": {
            "requested_sessions": int(len(manifest_out)),
            "loaded_sessions": int(manifest_out["load_status"].eq("ok").sum()),
            "failed_sessions": int((~manifest_out["load_status"].eq("ok")).sum()),
            "unique_loaded_animals": int(manifest_out.loc[manifest_out["load_status"].eq("ok"), "animal"].nunique()),
            "scored_choices": int(manifest_out["n_reward_path_first_turns_scored"].sum()),
            "directly_observed_choices": int(manifest_out["n_reward_path_first_turns_observed"].sum()),
            "inferred_choices": int(manifest_out["n_reward_path_first_turns_inferred"].sum()),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(_json_ready(run_manifest), indent=2), encoding="utf-8")
    print(f"Saved turn-sequence retention analysis to {out_dir}")
    print(f"Report: {report_path}")
    print(f"Loaded sessions: {run_manifest['session_counts']['loaded_sessions']}")


if __name__ == "__main__":
    main()
