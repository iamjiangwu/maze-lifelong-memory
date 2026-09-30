"""Frozen ED2 renderer adapter for a common depth-supported panel-i cohort.

Round 26 preserves the Round 25 renderer and all natural simulation rows.  It
changes only the panel-i display cohort: the depth-varying model is evaluated
on the exact all-depth-strata support used by panels g and h (at least 10
P_BS events in each of the shallow, middle, and deep strata).  The four-policy
panel f remains on its natural 23/58/74 cohort.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "figure_version_records").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
BASE_RENDERER = (
    REPO_ROOT
    / "docs"
    / "figure_version_records_archive_2026-08-09"
    / "extended_data_fig02_2026-08-12_round23_preceding_turn"
    / "source_code"
    / "render_ed03_fixed_time_30min_full_story.py"
)
SPEC = importlib.util.spec_from_file_location("ed02_round23_renderer", BASE_RENDERER)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_RENDERER)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

for NAME in dir(BASE):
    if not NAME.startswith("__"):
        globals()[NAME] = getattr(BASE, NAME)


FOUR_COUNTS = {"naive": 23, "LTM": 58, "experienced": 74}
DEPTH_COUNTS = {"naive": 22, "LTM": 58, "experienced": 74}
DEPTH_SUPPORT_COLUMNS = ("n_Lo_shallow", "n_Lo_mid", "n_Lo_deep")
MIN_DEPTH_EVENTS = 10


def _depth_supported(frame):
    support = BASE.pd.Series(True, index=frame.index)
    for column in DEPTH_SUPPORT_COLUMNS:
        support &= BASE.pd.to_numeric(frame[column], errors="coerce").ge(
            MIN_DEPTH_EVENTS
        )
    return support


def validate_source_cohorts(frames):
    strict_counts = frames["strict_policy_rows"].groupby("group").size().to_dict()
    if strict_counts != {"LTM": 58, "naive": 23}:
        raise ValueError(f"Four-policy cohort drifted: {strict_counts}")
    natural_counts = frames["natural_policy_rows"].groupby("group").size().to_dict()
    if natural_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural cohort drifted: {natural_counts}")
    simulation = frames["natural_simulation_rows"]
    simulation_counts = simulation.groupby("group").size().to_dict()
    if simulation_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural calibration cohort drifted: {simulation_counts}")
    depth = simulation.loc[_depth_supported(simulation)]
    depth_counts = depth.groupby("group").size().to_dict()
    if depth_counts != {"LTM": 58, "experienced": 74, "naive": 22}:
        raise ValueError(f"Depth-supported calibration cohort drifted: {depth_counts}")
    for key, frame in frames.items():
        if not BASE.pd.to_numeric(frame["window_minutes"], errors="coerce").eq(30).all():
            raise ValueError(f"Non-30-minute row survived in {key}")


def prepare_calibration(session_frame, summary_frame, model):
    if model not in {"four_bias", "depth_stratified"}:
        raise ValueError(model)
    simulated_column = (
        "four_own_half_cover_eff"
        if model == "four_bias"
        else "depth_own_half_cover_eff"
    )
    points = session_frame.loc[session_frame["group"].isin(BASE.GROUP_ORDER)].copy()
    if model == "depth_stratified":
        points = points.loc[_depth_supported(points)].copy()
    points = BASE.numeric(points, ("real_half_cover_eff", simulated_column))
    points = points.dropna(
        subset=["real_half_cover_eff", simulated_column]
    ).reset_index(drop=True)
    expected = FOUR_COUNTS if model == "four_bias" else DEPTH_COUNTS
    observed = points.groupby("group").size().to_dict()
    if observed != expected:
        raise ValueError(f"{model} calibration rows drifted: {observed}")

    summary = summary_frame.copy()
    summary["model_norm"] = summary["model"].map(BASE.normalize_model)
    summary = summary.loc[
        summary["model_norm"].eq(model)
        & summary["group"].isin(BASE.GROUP_ORDER)
    ].copy()
    summary = BASE.numeric(summary, ("mean_error", "pearson_r", "mae", "slope"))
    if len(summary) != 3:
        raise ValueError(f"{model} calibration requires three summary rows")
    manuscript_columns = [
        "window_minutes",
        "model",
        "group",
        "group_label",
        "n_session_rows",
        "n_mice",
        "mean_error",
        "mae",
        "pearson_r",
        "slope",
        "real_mean",
        "simulated_mean",
        "n_zero_simulated_rows",
        "model_norm",
    ]
    return points, summary[manuscript_columns].reset_index(drop=True), simulated_column


BASE.validate_source_cohorts = validate_source_cohorts
BASE.prepare_calibration = prepare_calibration
load_sources = BASE.load_sources
