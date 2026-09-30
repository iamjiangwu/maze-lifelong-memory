"""Duration-matched maze vs outside-maze ensemble-rate correlation control.

Supports the historical fixed-duration analysis and a context-only analysis
that uses every calcium-valid outside-maze frame in each session and draws the
same session-specific number of in-maze frames. The latter removes the
position/speed eligibility gate from the in-maze pool while retaining matched
duration within each session. Both modes recompute C-signal ensemble-rate
correlations for the same matched-neuron sets and perform session/pair-level
OLS tests with standard errors clustered by animal. Animal-level paired tests
are saved as secondary summaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ca_analysis.config import DT, FPS, get_behav_wd, get_neural_wd
from ca_analysis.io import load_behav_neural
from ca_analysis.trajectory import LoadTraj
from scripts.analysis_common import VARIANT_ORDER, make_audit_dir, resolve_repo_path, save_summary_json
from scripts.analyze_home_cage_rate_control import VARIANT_COLORS, _normalize_join_keys, _t_ci_1sample


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pair-summary",
        default="docs/cohort_update_839_842_2026-04-03/pv_pair_summary_merged.csv",
    )
    p.add_argument(
        "--maze-rate-path",
        default="docs/firing_rate_preservation_C_merged_cohort_2026-04-23/rate_preservation_per_pair.csv",
    )
    p.add_argument("--behav-wd", default=get_behav_wd())
    p.add_argument("--neural-wd", default=get_neural_wd())
    p.add_argument("--audit-dir", default="docs/home_cage_rate_control_equal_time_2026-05-14")
    p.add_argument("--target-min", type=float, default=5.0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--maze-frame-policy",
        choices=["selected_behavior", "annotated_context"],
        default="selected_behavior",
        help=(
            "selected_behavior uses the position/speed-filtered behavior-aligned rows; "
            "annotated_context uses every calcium-valid frame in saved in-maze intervals."
        ),
    )
    p.add_argument(
        "--sampling-policy",
        choices=["fixed_duration", "match_all_outside"],
        default="fixed_duration",
        help=(
            "fixed_duration samples --target-min from every pool; match_all_outside uses "
            "all outside-maze frames in each session and draws the same number in maze."
        ),
    )
    p.add_argument("--max-groups", type=int, default=None,
                   help="Optional smoke limit over group/animal quartets.")
    p.add_argument("--animals", nargs="*", default=None,
                   help="Optional animal-id filter for sharded runs.")
    p.add_argument("--checkpoint-each-group", action="store_true",
                   help="Write partial repeat/feasibility/failure CSVs after each animal group.")
    p.add_argument("--resume-from-partial", action="store_true",
                   help="Resume from partial CSVs written by --checkpoint-each-group.")
    p.add_argument("--chunk-neurons", type=int, default=256)
    return p.parse_args()


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    text = "|".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little", signed=False))


def _annotated_context_frames(session_id: str, total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    tf = LoadTraj(session_id + "-tf")
    fr = np.asarray(tf.fr, dtype=int)
    maze_mask = np.zeros(total_frames, dtype=bool)
    for start, end in fr:
        if end >= 0 and start < total_frames:
            maze_mask[max(start, 0):min(end + 1, total_frames)] = True
    return np.flatnonzero(maze_mask), np.flatnonzero(~maze_mask)


def _load_session(
    session_id: str,
    behav_wd: str,
    neural_wd: str,
    *,
    maze_frame_policy: str,
) -> dict[str, Any]:
    _tr, selected_df, _nose, _c_raw, c_mat, s_mat, _cn = load_behav_neural(
        session_id, behav_wd=behav_wd, neural_wd=neural_wd,
    )
    if c_mat is None or s_mat is None:
        raise RuntimeError(f"Missing C or S for {session_id}")
    if c_mat.shape[1] != s_mat.shape[1]:
        raise RuntimeError(f"C/S frame mismatch for {session_id}: {c_mat.shape[1]} != {s_mat.shape[1]}")
    total_frames = int(c_mat.shape[1])
    annotated_maze_idx, outside_idx = _annotated_context_frames(session_id, total_frames)
    neural_frame_valid = np.isfinite(c_mat[0])
    outside_idx = outside_idx[neural_frame_valid[outside_idx]]
    annotated_maze_idx = annotated_maze_idx[neural_frame_valid[annotated_maze_idx]]
    selected_maze_idx = selected_df["frame"].to_numpy(dtype=int)
    selected_maze_idx = selected_maze_idx[
        (selected_maze_idx >= 0) & (selected_maze_idx < total_frames)
    ]
    selected_maze_idx = selected_maze_idx[neural_frame_valid[selected_maze_idx]]
    if maze_frame_policy == "selected_behavior":
        maze_idx = selected_maze_idx
    elif maze_frame_policy == "annotated_context":
        maze_idx = annotated_maze_idx
    else:  # pragma: no cover - argparse constrains this in normal use.
        raise ValueError(f"Unknown maze frame policy: {maze_frame_policy}")
    return {
        "session_id": session_id,
        "c_mat": c_mat,
        "n_neurons": int(c_mat.shape[0]),
        "n_frames": total_frames,
        "home_idx": outside_idx,
        "maze_idx": maze_idx,
        "selected_maze_idx": selected_maze_idx,
        "annotated_maze_idx": annotated_maze_idx,
        "home_minutes": float(outside_idx.size * DT / 60.0),
        "maze_minutes": float(maze_idx.size * DT / 60.0),
        "selected_maze_minutes": float(selected_maze_idx.size * DT / 60.0),
        "annotated_maze_minutes": float(annotated_maze_idx.size * DT / 60.0),
        "n_neural_valid_frames": int(neural_frame_valid.sum()),
    }


def _rates_from_frames(
    c_mat: np.ndarray,
    frame_idx: np.ndarray,
    neuron_ids: np.ndarray,
    *,
    chunk_neurons: int,
) -> np.ndarray:
    dt_min = max(frame_idx.size * DT / 60.0, 1e-12)
    rates = np.empty(neuron_ids.size, dtype=float)
    for start in range(0, neuron_ids.size, chunk_neurons):
        stop = min(start + chunk_neurons, neuron_ids.size)
        ids = neuron_ids[start:stop]
        block = c_mat[np.ix_(ids, frame_idx)]
        rates[start:stop] = np.nansum(block, axis=1) / dt_min
    return rates


def _rate_corr(rates_train: np.ndarray, rates_test: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(rates_train) & np.isfinite(rates_test)
    ok &= (rates_train > 0) | (rates_test > 0)
    n_used = int(ok.sum())
    if n_used < 10:
        return np.nan, n_used
    r, _p = stats.pearsonr(rates_train[ok], rates_test[ok])
    return float(r), n_used


def _paired_tests(animal_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, sub in animal_df.groupby("variant", observed=True):
        diff = sub["equal_time_maze_minus_outside"].dropna()
        if diff.size < 3:
            continue
        t_stat, p_t = stats.ttest_1samp(diff, 0.0)
        try:
            w_stat, p_w = stats.wilcoxon(diff)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        ci_low, ci_high = _t_ci_1sample(diff)
        rows.append({
            "test_family": "equal_time_maze_minus_outside",
            "variant": variant,
            "n_animals": int(diff.size),
            "mean_diff": float(diff.mean()),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "t_stat": float(t_stat),
            "t_pvalue": float(p_t),
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_pvalue": float(p_w) if np.isfinite(p_w) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["t_qvalue_bh"] = multipletests(out["t_pvalue"], method="fdr_bh")[1]
    return out


def _clustered_pair_tests(pair_df: pd.DataFrame, metric: str, *, family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, sub in pair_df.groupby("variant", observed=True):
        n_animals = int(sub["animal_id"].nunique())
        if n_animals >= 2:
            model = smf.ols(f"{metric} ~ 1", data=sub).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": sub["animal_id"],
                    "use_correction": True,
                    "df_correction": True,
                },
            )
            cov_type = "cluster"
            cluster_df = n_animals - 1
        else:
            model = smf.ols(f"{metric} ~ 1", data=sub).fit(cov_type="HC3")
            cov_type = "HC3_single_animal_smoke"
            cluster_df = np.nan
        t_stat = float(model.tvalues["Intercept"])
        pvalue = (
            float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=cluster_df)))
            if np.isfinite(t_stat) and np.isfinite(cluster_df) and cluster_df > 0
            else float(model.pvalues["Intercept"])
        )
        rows.append({
            "test_family": family,
            "variant": variant,
            "term": "Intercept",
            "n_pair_rows": int(len(sub)),
            "n_animals": n_animals,
            "cluster_df": cluster_df,
            "mean_diff": float(model.params["Intercept"]),
            "cluster_se": float(model.bse["Intercept"]),
            "t_stat": t_stat,
            "pvalue": pvalue,
            "model_pvalue": float(model.pvalues["Intercept"]),
            "cov_type": cov_type,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["qvalue_bh"] = multipletests(out["pvalue"], method="fdr_bh")[1]
    return out


def _specificity_tests(animal_df: pd.DataFrame) -> pd.DataFrame:
    base = animal_df[animal_df["variant"] == "d1v1"].set_index("animal_id")[
        "equal_time_maze_minus_outside"
    ]
    rows: list[dict[str, Any]] = []
    for variant in ["d1v2", "d2v1", "d2v2"]:
        other = animal_df[animal_df["variant"] == variant].set_index("animal_id")[
            "equal_time_maze_minus_outside"
        ]
        common = base.index.intersection(other.index)
        diff = (base.loc[common] - other.loc[common]).dropna()
        if diff.size < 3:
            continue
        t_stat, p_t = stats.ttest_1samp(diff, 0.0)
        ci_low, ci_high = _t_ci_1sample(diff)
        rows.append({
            "test_family": "equal_time_delta_specificity",
            "comparison": f"d1v1_minus_{variant}",
            "n_animals": int(diff.size),
            "mean_diff": float(diff.mean()),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "t_stat": float(t_stat),
            "t_pvalue": float(p_t),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["t_qvalue_bh"] = multipletests(out["t_pvalue"], method="fdr_bh")[1]
    return out


def _write_figures(audit_dir: Path, animal_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, variant in zip(axes, VARIANT_ORDER):
        sub = animal_df[animal_df["variant"] == variant]
        color = VARIANT_COLORS[variant]
        for _, row in sub.iterrows():
            ax.plot(
                [0, 1],
                [row["equal_time_outside_r"], row["equal_time_maze_r"]],
                color=color,
                alpha=0.35,
                linewidth=1.0,
            )
            ax.scatter(
                [0, 1],
                [row["equal_time_outside_r"], row["equal_time_maze_r"]],
                color=color,
                alpha=0.55,
                s=18,
            )
        means = [sub["equal_time_outside_r"].mean(), sub["equal_time_maze_r"].mean()]
        sems = [
            0.0 if not np.isfinite(sub["equal_time_outside_r"].sem()) else sub["equal_time_outside_r"].sem(),
            0.0 if not np.isfinite(sub["equal_time_maze_r"].sem()) else sub["equal_time_maze_r"].sem(),
        ]
        ax.errorbar([0, 1], means, yerr=sems, color="black", marker="o", linewidth=1.2, capsize=3)
        ax.set_xticks([0, 1], ["Outside", "Maze"])
        ax.set_title(variant)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Equal-time ensemble-rate correlation")
    fig.tight_layout()
    fig.savefig(audit_dir / "equal_time_maze_outside_paired_lines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, variant in enumerate(VARIANT_ORDER):
        sub = animal_df[animal_df["variant"] == variant]
        color = VARIANT_COLORS[variant]
        ax.bar(i, sub["equal_time_maze_minus_outside"].mean(), color=color, alpha=0.7,
               edgecolor="black", linewidth=0.6)
        sem = sub["equal_time_maze_minus_outside"].sem()
        ax.errorbar(i, sub["equal_time_maze_minus_outside"].mean(),
                    yerr=0.0 if not np.isfinite(sem) else sem, color="black", capsize=3)
        ax.scatter(np.full(len(sub), i), sub["equal_time_maze_minus_outside"],
                   color="black", alpha=0.45, s=16)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xticks(np.arange(len(VARIANT_ORDER)), VARIANT_ORDER)
    ax.set_ylabel("Equal-time maze - outside")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(audit_dir / "equal_time_delta_by_variant.png", dpi=180)
    plt.close(fig)


def _read_partial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _write_partial_tables(
    audit_dir: Path,
    rows: list[dict[str, Any]],
    pair_audits: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(audit_dir / "equal_time_repeat_results.partial.csv", index=False)
    if pair_audits:
        pd.DataFrame(pair_audits).to_csv(audit_dir / "equal_time_pair_feasibility.partial.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(audit_dir / "equal_time_failures.partial.csv", index=False)


def main() -> None:
    args = _parse_args()
    audit_dir = make_audit_dir(args.audit_dir)
    pair_df = _normalize_join_keys(pd.read_csv(resolve_repo_path(args.pair_summary)))
    maze_df = _normalize_join_keys(pd.read_csv(resolve_repo_path(args.maze_rate_path)))
    keys = ["variant", "group", "animal_id", "src_session", "dst_session"]
    pair_keys = pair_df[keys].merge(maze_df[keys], on=keys, how="outer", indicator=True)
    pair_keys.to_csv(audit_dir / "equal_time_join_key_audit_rows.csv", index=False)
    join_audit = {
        "pair_summary_rows": int(len(pair_df)),
        "maze_rows": int(len(maze_df)),
        "inner_join_rows": int((pair_keys["_merge"] == "both").sum()),
        "pair_only_rows": int((pair_keys["_merge"] == "left_only").sum()),
        "maze_only_rows": int((pair_keys["_merge"] == "right_only").sum()),
        "target_minutes": (
            float(args.target_min) if args.sampling_policy == "fixed_duration" else None
        ),
        "target_frames": (
            int(round(args.target_min * 60.0 * FPS))
            if args.sampling_policy == "fixed_duration"
            else None
        ),
        "fixed_duration_parameter_minutes": float(args.target_min),
        "n_repeats": int(args.n_repeats),
        "maze_frame_policy": str(args.maze_frame_policy),
        "sampling_policy": str(args.sampling_policy),
        "frame_timebase": "original neural imaging-frame index",
        "common_neural_frame_gate": "finite first C trace in both contexts",
    }
    pd.DataFrame([join_audit]).to_csv(audit_dir / "equal_time_join_audit.csv", index=False)
    rows: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_frames = int(round(args.target_min * 60.0 * FPS))

    grouped = list(pair_df.groupby(["group", "animal_id"], sort=False))
    if args.animals:
        animal_filter = {str(animal) for animal in args.animals}
        grouped = [
            ((group, animal_id), group_df)
            for (group, animal_id), group_df in grouped
            if str(animal_id) in animal_filter
        ]
    if args.max_groups is not None:
        grouped = grouped[:args.max_groups]

    completed_groups: set[tuple[str, str]] = set()
    if args.resume_from_partial:
        rows = _read_partial_rows(audit_dir / "equal_time_repeat_results.partial.csv")
        pair_audits = _read_partial_rows(audit_dir / "equal_time_pair_feasibility.partial.csv")
        failures = _read_partial_rows(audit_dir / "equal_time_failures.partial.csv")
        if rows:
            partial_repeat = pd.DataFrame(rows)
            for (group, animal_id), group_df in grouped:
                sub = partial_repeat[
                    (partial_repeat["group"].astype(str) == str(group))
                    & (partial_repeat["animal_id"].astype(str) == str(animal_id))
                ]
                expected_rows = len(group_df) * int(args.n_repeats)
                if len(sub) >= expected_rows:
                    completed_groups.add((str(group), str(animal_id)))
            if completed_groups:
                print(
                    f"Resuming from partial rows; skipping {len(completed_groups)} completed groups.",
                    flush=True,
                )

    for group_i, ((group, animal_id), group_df) in enumerate(grouped, start=1):
        if (str(group), str(animal_id)) in completed_groups:
            print(f"[{group_i}/{len(grouped)}] {group} {animal_id}: skipped from partial", flush=True)
            continue
        sessions = sorted(set(group_df["src_session"]).union(set(group_df["dst_session"])))
        print(f"[{group_i}/{len(grouped)}] {group} {animal_id}: {len(sessions)} sessions", flush=True)
        session_data: dict[str, dict[str, Any]] = {}
        try:
            for session_id in sessions:
                session_data[session_id] = _load_session(
                    session_id,
                    args.behav_wd,
                    args.neural_wd,
                    maze_frame_policy=args.maze_frame_policy,
                )
            for _, row in group_df.iterrows():
                variant = str(row["variant"])
                src = str(row["src_session"])
                dst = str(row["dst_session"])
                data_path = Path(str(row["data_path"]))
                try:
                    with data_path.open("rb") as fh:
                        payload = pickle.load(fh)
                    train_ids = np.asarray(payload["train_ids"], dtype=int)
                    test_ids = np.asarray(payload["test_ids"], dtype=int)
                    if train_ids.size != test_ids.size:
                        raise ValueError(f"ID length mismatch {train_ids.size}/{test_ids.size}")
                    src_dat = session_data[src]
                    dst_dat = session_data[dst]
                    if train_ids.min(initial=0) < 0 or train_ids.max(initial=-1) >= src_dat["n_neurons"]:
                        raise ValueError("train ID out of bounds")
                    if test_ids.min(initial=0) < 0 or test_ids.max(initial=-1) >= dst_dat["n_neurons"]:
                        raise ValueError("test ID out of bounds")
                    pool_sizes = {
                        "outside_train_frames": int(src_dat["home_idx"].size),
                        "outside_test_frames": int(dst_dat["home_idx"].size),
                        "maze_train_frames": int(src_dat["maze_idx"].size),
                        "maze_test_frames": int(dst_dat["maze_idx"].size),
                        "selected_maze_train_frames": int(src_dat["selected_maze_idx"].size),
                        "selected_maze_test_frames": int(dst_dat["selected_maze_idx"].size),
                        "annotated_maze_train_frames": int(src_dat["annotated_maze_idx"].size),
                        "annotated_maze_test_frames": int(dst_dat["annotated_maze_idx"].size),
                    }
                    if args.sampling_policy == "fixed_duration":
                        draw_sizes = {
                            "outside_train_draw_frames": target_frames,
                            "outside_test_draw_frames": target_frames,
                            "maze_train_draw_frames": target_frames,
                            "maze_test_draw_frames": target_frames,
                        }
                        feasible = min(
                            pool_sizes["outside_train_frames"],
                            pool_sizes["outside_test_frames"],
                            pool_sizes["maze_train_frames"],
                            pool_sizes["maze_test_frames"],
                        ) >= target_frames
                    elif args.sampling_policy == "match_all_outside":
                        draw_sizes = {
                            "outside_train_draw_frames": pool_sizes["outside_train_frames"],
                            "outside_test_draw_frames": pool_sizes["outside_test_frames"],
                            "maze_train_draw_frames": pool_sizes["outside_train_frames"],
                            "maze_test_draw_frames": pool_sizes["outside_test_frames"],
                        }
                        feasible = (
                            draw_sizes["outside_train_draw_frames"] > 0
                            and draw_sizes["outside_test_draw_frames"] > 0
                            and pool_sizes["maze_train_frames"] >= draw_sizes["maze_train_draw_frames"]
                            and pool_sizes["maze_test_frames"] >= draw_sizes["maze_test_draw_frames"]
                        )
                    else:  # pragma: no cover - argparse constrains this in normal use.
                        raise ValueError(f"Unknown sampling policy: {args.sampling_policy}")
                    pair_audits.append({
                        "variant": variant,
                        "group": str(group),
                        "animal_id": str(animal_id),
                        "src_session": src,
                        "dst_session": dst,
                        "target_frames": (
                            target_frames if args.sampling_policy == "fixed_duration" else np.nan
                        ),
                        "target_minutes": (
                            float(args.target_min)
                            if args.sampling_policy == "fixed_duration"
                            else np.nan
                        ),
                        "feasible": bool(feasible),
                        "n_matched": int(train_ids.size),
                        "maze_frame_policy": str(args.maze_frame_policy),
                        "sampling_policy": str(args.sampling_policy),
                        "selected_train_fraction_of_annotated_maze": (
                            pool_sizes["selected_maze_train_frames"]
                            / max(pool_sizes["annotated_maze_train_frames"], 1)
                        ),
                        "selected_test_fraction_of_annotated_maze": (
                            pool_sizes["selected_maze_test_frames"]
                            / max(pool_sizes["annotated_maze_test_frames"], 1)
                        ),
                        **draw_sizes,
                        **pool_sizes,
                    })
                    if not feasible:
                        failures.append({
                            "variant": variant, "group": str(group), "animal_id": str(animal_id),
                            "src_session": src, "dst_session": dst, "error": "insufficient context frames",
                            "maze_frame_policy": str(args.maze_frame_policy),
                            "sampling_policy": str(args.sampling_policy),
                            **draw_sizes,
                            **pool_sizes,
                        })
                        continue
                    if args.sampling_policy == "match_all_outside":
                        h_tr = src_dat["home_idx"]
                        h_te = dst_dat["home_idx"]
                        h_rates_tr = _rates_from_frames(
                            src_dat["c_mat"], h_tr, train_ids, chunk_neurons=args.chunk_neurons,
                        )
                        h_rates_te = _rates_from_frames(
                            dst_dat["c_mat"], h_te, test_ids, chunk_neurons=args.chunk_neurons,
                        )
                        fixed_h_r, fixed_h_n = _rate_corr(h_rates_tr, h_rates_te)
                    for repeat in range(args.n_repeats):
                        rng = _stable_rng(args.seed, group, animal_id, variant, src, dst, repeat)
                        if args.sampling_policy == "fixed_duration":
                            h_tr = np.sort(rng.choice(
                                src_dat["home_idx"],
                                size=draw_sizes["outside_train_draw_frames"],
                                replace=False,
                            ))
                            h_te = np.sort(rng.choice(
                                dst_dat["home_idx"],
                                size=draw_sizes["outside_test_draw_frames"],
                                replace=False,
                            ))
                            h_rates_tr = _rates_from_frames(
                                src_dat["c_mat"], h_tr, train_ids, chunk_neurons=args.chunk_neurons,
                            )
                            h_rates_te = _rates_from_frames(
                                dst_dat["c_mat"], h_te, test_ids, chunk_neurons=args.chunk_neurons,
                            )
                            h_r, h_n = _rate_corr(h_rates_tr, h_rates_te)
                        else:
                            h_r, h_n = fixed_h_r, fixed_h_n
                        m_tr = np.sort(rng.choice(
                            src_dat["maze_idx"],
                            size=draw_sizes["maze_train_draw_frames"],
                            replace=False,
                        ))
                        m_te = np.sort(rng.choice(
                            dst_dat["maze_idx"],
                            size=draw_sizes["maze_test_draw_frames"],
                            replace=False,
                        ))
                        m_rates_tr = _rates_from_frames(
                            src_dat["c_mat"], m_tr, train_ids, chunk_neurons=args.chunk_neurons,
                        )
                        m_rates_te = _rates_from_frames(
                            dst_dat["c_mat"], m_te, test_ids, chunk_neurons=args.chunk_neurons,
                        )
                        m_r, m_n = _rate_corr(m_rates_tr, m_rates_te)
                        rows.append({
                            "variant": variant,
                            "group": str(group),
                            "animal_id": str(animal_id),
                            "src_session": src,
                            "dst_session": dst,
                            "repeat": repeat,
                            "target_minutes": (
                                float(args.target_min)
                                if args.sampling_policy == "fixed_duration"
                                else np.nan
                            ),
                            "target_frames": (
                                target_frames if args.sampling_policy == "fixed_duration" else np.nan
                            ),
                            "maze_frame_policy": str(args.maze_frame_policy),
                            "sampling_policy": str(args.sampling_policy),
                            **draw_sizes,
                            "equal_time_outside_r": h_r,
                            "equal_time_maze_r": m_r,
                            "equal_time_maze_minus_outside": m_r - h_r if np.isfinite(h_r) and np.isfinite(m_r) else np.nan,
                            "equal_time_outside_n_used": h_n,
                            "equal_time_maze_n_used": m_n,
                        })
                except Exception as exc:
                    failures.append({
                        "variant": variant,
                        "group": str(group),
                        "animal_id": str(animal_id),
                        "src_session": src,
                        "dst_session": dst,
                        "error": str(exc),
                    })
        finally:
            session_data.clear()
            gc.collect()
            if args.checkpoint_each_group:
                _write_partial_tables(audit_dir, rows, pair_audits, failures)

    repeat_df = pd.DataFrame(rows)
    repeat_df.to_csv(audit_dir / "equal_time_repeat_results.csv", index=False)
    pair_audit_df = pd.DataFrame(pair_audits)
    pair_audit_df.to_csv(audit_dir / "equal_time_pair_feasibility.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(audit_dir / "equal_time_failures.csv", index=False)
    if repeat_df.empty:
        raise RuntimeError("No equal-time repeat rows were produced.")

    pair_summary = (
        repeat_df.groupby(keys, observed=True)[
            ["equal_time_outside_r", "equal_time_maze_r", "equal_time_maze_minus_outside"]
        ]
        .agg(["mean", "std", "count"])
    )
    pair_summary.columns = ["_".join(c) for c in pair_summary.columns]
    pair_summary = pair_summary.reset_index()
    pair_summary.to_csv(audit_dir / "equal_time_pair_summary.csv", index=False)
    _clustered_pair_tests(
        pair_summary,
        "equal_time_maze_minus_outside_mean",
        family="equal_time_session_pair_clustered_maze_minus_outside",
    ).to_csv(audit_dir / "equal_time_clustered_pair_tests.csv", index=False)
    if pair_summary["animal_id"].nunique() >= 2:
        spec_model = smf.ols(
            "equal_time_maze_minus_outside_mean ~ C(variant)",
            data=pair_summary,
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": pair_summary["animal_id"],
                "use_correction": True,
                "df_correction": True,
            },
        )
    else:
        spec_model = smf.ols(
            "equal_time_maze_minus_outside_mean ~ C(variant)",
            data=pair_summary,
        ).fit(cov_type="HC3")
    cluster_df = (
        int(pair_summary["animal_id"].nunique()) - 1
        if pair_summary["animal_id"].nunique() >= 2
        else np.nan
    )
    pd.DataFrame([
        {
            "term": term,
            "beta": float(spec_model.params[term]),
            "se": float(spec_model.bse[term]),
            "t_stat": float(spec_model.tvalues[term]),
            "pvalue": (
                float(2.0 * (1.0 - stats.t.cdf(abs(float(spec_model.tvalues[term])), df=cluster_df)))
                if np.isfinite(float(spec_model.tvalues[term])) and np.isfinite(cluster_df) and cluster_df > 0
                else float(spec_model.pvalues[term])
            ),
            "model_pvalue": float(spec_model.pvalues[term]),
            "cluster_df": cluster_df,
            "n_rows": int(spec_model.nobs),
            "r_squared": float(spec_model.rsquared),
            "cov_type": "cluster" if pair_summary["animal_id"].nunique() >= 2 else "HC3_single_animal_smoke",
        }
        for term in spec_model.params.index
    ]).to_csv(audit_dir / "equal_time_clustered_delta_specificity_model.csv", index=False)

    animal_df = (
        pair_summary.groupby(["animal_id", "variant"], observed=True)[
            [
                "equal_time_outside_r_mean",
                "equal_time_maze_r_mean",
                "equal_time_maze_minus_outside_mean",
            ]
        ]
        .mean()
        .reset_index()
        .rename(columns={
            "equal_time_outside_r_mean": "equal_time_outside_r",
            "equal_time_maze_r_mean": "equal_time_maze_r",
            "equal_time_maze_minus_outside_mean": "equal_time_maze_minus_outside",
        })
    )
    animal_df.to_csv(audit_dir / "equal_time_animal_level.csv", index=False)
    tests = _paired_tests(animal_df)
    tests.to_csv(audit_dir / "equal_time_animal_tests.csv", index=False)
    spec = _specificity_tests(animal_df)
    spec.to_csv(audit_dir / "equal_time_delta_specificity_tests.csv", index=False)
    _write_figures(audit_dir, animal_df)
    save_summary_json(
        audit_dir,
        {
            **join_audit,
            "repeat_rows": int(len(repeat_df)),
            "pair_rows_with_results": int(pair_summary.shape[0]),
            "n_animals": int(animal_df["animal_id"].nunique()),
            "variant_animal_counts": animal_df.groupby("variant")["animal_id"].nunique().to_dict(),
            "mean_equal_time_maze_minus_outside_by_variant": animal_df.groupby("variant")[
                "equal_time_maze_minus_outside"
            ].mean().to_dict(),
        },
    )
    print({
        "audit_dir": str(audit_dir),
        "repeat_rows": len(repeat_df),
        "pair_rows": len(pair_summary),
        "n_animals": animal_df["animal_id"].nunique(),
    })


if __name__ == "__main__":
    main()
