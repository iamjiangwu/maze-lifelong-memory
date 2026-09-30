from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.export_turn_sequence_retention import (
    _build_initial_path_reconstruction,
    _summarize_session_routes,
)
from scripts.render_ed02_ed04_current_panels import parse_required_binary


def _maze_for_entry_path() -> SimpleNamespace:
    n_nodes = 17
    parents = np.full(n_nodes, -1, dtype=int)
    parents[1] = 0
    parents[3] = 1
    parents[7] = 3
    parents[16] = 7
    return SimpleNamespace(
        ru=[[node] for node in range(n_nodes)],
        pa=parents,
        xc=np.arange(n_nodes, dtype=int),
        yc=np.zeros(n_nodes, dtype=int),
        rc=np.arange(n_nodes, dtype=int),
    )


def _reward_path() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_depth": 1,
                "parent_node": 0,
                "reward_child": 1,
                "sibling_child": 2,
                "left_child": 1,
                "right_child": 2,
            },
            {
                "paper_depth": 2,
                "parent_node": 1,
                "reward_child": 4,
                "sibling_child": 3,
                "left_child": 3,
                "right_child": 4,
            },
        ]
    )


def _session_row(session_id: str = "489-3day1") -> pd.Series:
    return pd.Series(
        {
            "session_id": session_id,
            "animal": "489",
            "condition": "return",
        }
    )


def test_approved_entry_path_reconstructs_every_crossed_reward_parent() -> None:
    maze = _maze_for_entry_path()
    trajectory = SimpleNamespace(
        fr=np.asarray([[378, 5000]], dtype=float),
        ce=[np.asarray([16, 16, 16], dtype=float)],
        no=[np.asarray([[16, 0]], dtype=float)],
    )

    reconstructed, audit, frame_audit = _build_initial_path_reconstruction(
        trajectory,
        maze,
        _session_row(),
        _reward_path(),
        timing={"first_reward_abs_frame": 4834.0},
    )

    assert reconstructed == {
        0: {"chosen_child": 1, "choice_side": "left", "reward_correct": True},
        1: {"chosen_child": 3, "choice_side": "left", "reward_correct": False},
    }
    assert audit.loc[0, "home_to_entry_run_path"] == "0;1;3;7;16"
    assert audit.loc[0, "home_to_entry_cell_path"] == "0;1;3;7;16"
    assert audit.loc[0, "first_valid_abs_frame"] == 378
    assert audit.loc[0, "n_reconstructed_reward_parent_choices"] == 2
    assert frame_audit.loc[frame_audit["is_first_valid_sample"], "cell"].tolist() == [16]


def test_approved_entry_path_fails_closed_on_cell_run_disagreement() -> None:
    maze = _maze_for_entry_path()
    trajectory = SimpleNamespace(
        fr=np.asarray([[378, 5000]], dtype=float),
        ce=[np.asarray([16], dtype=float)],
        no=[np.asarray([[15, 0]], dtype=float)],
    )

    with pytest.raises(ValueError, match="failed closed"):
        _build_initial_path_reconstruction(
            trajectory,
            maze,
            _session_row(),
            _reward_path(),
            timing={"first_reward_abs_frame": 4834.0},
        )


def test_unapproved_session_is_not_reconstructed() -> None:
    maze = _maze_for_entry_path()
    trajectory = SimpleNamespace(
        fr=np.asarray([[378, 5000]], dtype=float),
        ce=[np.asarray([16], dtype=float)],
        no=[np.asarray([[16, 0]], dtype=float)],
    )

    reconstructed, audit, _ = _build_initial_path_reconstruction(
        trajectory,
        maze,
        _session_row("other-day1"),
        _reward_path(),
        timing={"first_reward_abs_frame": 4834.0},
    )

    assert reconstructed == {}
    assert audit.loc[0, "status"] == "not_approved"


def test_route_summary_separates_observed_inferred_and_scored_counts() -> None:
    correctness = [True, False, False, True, True, True]
    origins = ["inferred", "inferred", "observed", "observed", "observed", "observed"]
    rows = []
    for paper_depth, (correct, origin) in enumerate(zip(correctness, origins), start=1):
        rows.append(
            {
                "session_id": "489-3day1",
                "animal": "489",
                "condition": "return",
                "condition_label": "Return D1",
                "memory_stage": "3mo",
                "cohort_key": "LTMThreeDay1",
                "paper_depth": paper_depth,
                "reward_side": "left" if paper_depth in {1, 4, 6} else "right",
                "required_transition_type": "start" if paper_depth == 1 else "alternate_required",
                "previous_reward_side": "",
                "chosen_child": paper_depth,
                "reward_child": paper_depth if correct else paper_depth + 100,
                "sibling_child": paper_depth + 200,
                "first_reward_maze_seconds": 178.24,
                "reward_correct": correct,
                "choice_origin": origin,
                "directly_observed": origin == "observed",
            }
        )

    _, _, session_routes = _summarize_session_routes(pd.DataFrame(rows))
    route = session_routes.iloc[0]

    assert route["n_decisions_scored"] == 6
    assert route["n_decisions_observed"] == 4
    assert route["n_decisions_inferred"] == 2
    assert route["n_correct"] == 4
    assert route["route_fraction_correct"] == pytest.approx(4 / 6)
    assert bool(route["all6_observed"]) is False
    assert bool(route["all6_resolved"]) is True
    assert bool(route["exact_six_turn_sequence"]) is False


def test_renderer_boolean_parser_fails_closed_on_missing_choice() -> None:
    assert parse_required_binary(True) is True
    assert parse_required_binary("False") is False
    with pytest.raises(ValueError, match="Unresolved or invalid"):
        parse_required_binary(np.nan)
