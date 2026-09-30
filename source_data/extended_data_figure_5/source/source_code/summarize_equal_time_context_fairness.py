"""Combine and audit the context-fair ensemble-rate sensitivity shards."""

from __future__ import annotations

import argparse
import hashlib
import json
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


VARIANT_ORDER = ["d1v1", "d1v2", "d2v1", "d2v2"]
VARIANT_LABELS = ["D1-D1", "D1-D2", "D2-D1", "D2-D2"]
PAIR_KEYS = ["variant", "group", "animal_id", "src_session", "dst_session"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--accepted-pair-summary",
        type=Path,
        default=Path(
            "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
            "ed06_equal_time_pair_summary_corrected36_all_variants.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/equal_time_context_fairness_2026-08-20/combined_primary"),
    )
    parser.add_argument("--definition-label", default="Context-only, all outside")
    parser.add_argument("--output-tag", default="context_only_all_outside")
    parser.add_argument(
        "--sampling-rule-text",
        default=(
            "all calcium-valid outside-maze frames in each session; same session-specific "
            "number drawn without replacement from annotated in-maze frames"
        ),
    )
    parser.add_argument(
        "--allow-partial-outside",
        action="store_true",
        help="Allow a fixed-duration sensitivity that does not use every outside-maze frame.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.6g}" if np.isfinite(value) else "NA")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def read_shards(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    paths = [directory / filename for directory in run_dirs]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard outputs for {filename}: {missing}")
    frames = [pd.read_csv(path, dtype={"animal_id": str}) for path in paths]
    return pd.concat(frames, ignore_index=True)


def summarize_repeats(repeats: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "equal_time_outside_r",
        "equal_time_maze_r",
        "equal_time_maze_minus_outside",
    ]
    pair_summary = repeats.groupby(PAIR_KEYS, observed=True)[metrics].agg(["mean", "std", "count"])
    pair_summary.columns = ["_".join(column) for column in pair_summary.columns]
    return pair_summary.reset_index()


def clustered_intercepts(pair_summary: pd.DataFrame, *, definition: str) -> pd.DataFrame:
    metric_map = {
        "outside": "equal_time_outside_r_mean",
        "maze": "equal_time_maze_r_mean",
        "maze_minus_outside": "equal_time_maze_minus_outside_mean",
    }
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        subset = pair_summary.loc[pair_summary["variant"].eq(variant)].copy()
        n_animals = int(subset["animal_id"].nunique())
        cluster_df = n_animals - 1
        for estimand, metric in metric_map.items():
            model = smf.ols(f"{metric} ~ 1", data=subset).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": subset["animal_id"],
                    "use_correction": True,
                    "df_correction": True,
                },
            )
            estimate = float(model.params["Intercept"])
            se = float(model.bse["Intercept"])
            t_value = float(model.tvalues["Intercept"])
            p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_value), df=cluster_df)))
            critical = float(stats.t.ppf(0.975, df=cluster_df))
            rows.append(
                {
                    "definition": definition,
                    "variant": variant,
                    "estimand": estimand,
                    "n_pair_rows": int(len(subset)),
                    "n_animals": n_animals,
                    "cluster_df": cluster_df,
                    "estimate": estimate,
                    "cluster_se": se,
                    "ci95_low": estimate - critical * se,
                    "ci95_high": estimate + critical * se,
                    "t_stat": t_value,
                    "pvalue": p_value,
                }
            )
    output = pd.DataFrame(rows)
    is_delta = output["estimand"].eq("maze_minus_outside")
    output["pvalue_bh4"] = np.nan
    output.loc[is_delta, "pvalue_bh4"] = multipletests(
        output.loc[is_delta, "pvalue"], method="fdr_bh"
    )[1]
    return output


def definition_change_test(current: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    left = current[PAIR_KEYS + ["equal_time_maze_minus_outside_mean"]].rename(
        columns={"equal_time_maze_minus_outside_mean": "current_delta"}
    )
    right = primary[PAIR_KEYS + ["equal_time_maze_minus_outside_mean"]].rename(
        columns={"equal_time_maze_minus_outside_mean": "primary_delta"}
    )
    joined = left.merge(right, on=PAIR_KEYS, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Accepted and primary pair inventories do not match exactly")
    joined["primary_minus_current_delta"] = joined["primary_delta"] - joined["current_delta"]
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        subset = joined.loc[joined["variant"].eq(variant)].copy()
        n_animals = int(subset["animal_id"].nunique())
        model = smf.ols("primary_minus_current_delta ~ 1", data=subset).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": subset["animal_id"],
                "use_correction": True,
                "df_correction": True,
            },
        )
        cluster_df = n_animals - 1
        estimate = float(model.params["Intercept"])
        se = float(model.bse["Intercept"])
        critical = float(stats.t.ppf(0.975, df=cluster_df))
        t_value = float(model.tvalues["Intercept"])
        rows.append(
            {
                "variant": variant,
                "n_pair_rows": int(len(subset)),
                "n_animals": n_animals,
                "mean_primary_minus_current_delta": estimate,
                "cluster_se": se,
                "ci95_low": estimate - critical * se,
                "ci95_high": estimate + critical * se,
                "t_stat": t_value,
                "pvalue": float(2.0 * (1.0 - stats.t.cdf(abs(t_value), df=cluster_df))),
            }
        )
    tests = pd.DataFrame(rows)
    tests["pvalue_bh4"] = multipletests(tests["pvalue"], method="fdr_bh")[1]
    return joined, tests


def session_pool_audit(feasibility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side, session_col in [("train", "src_session"), ("test", "dst_session")]:
        for row in feasibility.itertuples(index=False):
            rows.append(
                {
                    "animal_id": str(row.animal_id),
                    "session_id": str(getattr(row, session_col)),
                    "outside_frames": int(getattr(row, f"outside_{side}_frames")),
                    "selected_maze_frames": int(getattr(row, f"selected_maze_{side}_frames")),
                    "annotated_maze_frames": int(getattr(row, f"annotated_maze_{side}_frames")),
                    "outside_draw_frames": int(getattr(row, f"outside_{side}_draw_frames")),
                    "maze_draw_frames": int(getattr(row, f"maze_{side}_draw_frames")),
                }
            )
    sessions = pd.DataFrame(rows).drop_duplicates()
    inconsistent = sessions.groupby(["animal_id", "session_id"], observed=True).size()
    if not inconsistent.eq(1).all():
        raise ValueError("A session has inconsistent pool sizes across pair rows")
    sessions["outside_minutes"] = sessions["outside_frames"] / 25.0 / 60.0
    sessions["selected_maze_minutes"] = sessions["selected_maze_frames"] / 25.0 / 60.0
    sessions["annotated_maze_minutes"] = sessions["annotated_maze_frames"] / 25.0 / 60.0
    sessions["matched_draw_minutes"] = sessions["maze_draw_frames"] / 25.0 / 60.0
    sessions["selected_fraction_of_annotated_maze"] = (
        sessions["selected_maze_frames"] / sessions["annotated_maze_frames"]
    )
    sessions["draw_counts_match_within_session"] = sessions["outside_draw_frames"].eq(
        sessions["maze_draw_frames"]
    )
    sessions["all_outside_frames_used"] = sessions["outside_draw_frames"].eq(
        sessions["outside_frames"]
    )
    return sessions.sort_values(["animal_id", "session_id"], kind="stable")


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def render_comparison(
    current: pd.DataFrame,
    primary: pd.DataFrame,
    stats_table: pd.DataFrame,
    joined: pd.DataFrame,
    out_dir: Path,
    *,
    primary_definition: str,
) -> list[Path]:
    configure_style()
    colors = {"Current filtered 5 min": "#9A9A9A", primary_definition: "#4A5E65"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    ax = axes[0]
    x_positions = {"outside": 0.0, "maze": 1.0}
    offsets = {"Current filtered 5 min": -0.09, primary_definition: 0.09}
    for definition in offsets:
        subset = stats_table.loc[
            stats_table["definition"].eq(definition)
            & stats_table["variant"].eq("d1v1")
            & stats_table["estimand"].isin(["outside", "maze"])
        ]
        for row in subset.itertuples(index=False):
            x = x_positions[row.estimand] + offsets[definition]
            ax.errorbar(
                x,
                row.estimate,
                yerr=[[row.estimate - row.ci95_low], [row.ci95_high - row.estimate]],
                fmt="o",
                color=colors[definition],
                capsize=2,
                label=definition if row.estimand == "outside" else None,
            )
    ax.set_xticks([0, 1], ["Outside", "Maze"])
    ax.set_ylabel("Ensemble-rate correlation (r)")
    ax.set_title("a  D1-D1 absolute values", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, loc="best")

    ax = axes[1]
    for definition, offset in offsets.items():
        subset = stats_table.loc[
            stats_table["definition"].eq(definition)
            & stats_table["estimand"].eq("maze_minus_outside")
        ].set_index("variant").loc[VARIANT_ORDER]
        x = np.arange(4, dtype=float) + offset
        y = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci95_low"].to_numpy(dtype=float)
        high = subset["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="o",
            color=colors[definition],
            capsize=2,
            label=definition,
        )
    ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(4), VARIANT_LABELS, rotation=28, ha="right")
    ax.set_ylabel("Maze - outside correlation (r)")
    ax.set_title("b  Duration-matched contrast", loc="left", fontweight="bold")

    ax = axes[2]
    d1 = joined.loc[joined["variant"].eq("d1v1")].copy()
    ax.scatter(d1["current_delta"], d1["primary_delta"], s=12, color="#4A5E65", alpha=0.75)
    bounds = np.array(
        [
            np.nanmin(d1[["current_delta", "primary_delta"]].to_numpy()),
            np.nanmax(d1[["current_delta", "primary_delta"]].to_numpy()),
        ]
    )
    padding = max(0.03, np.ptp(bounds) * 0.08)
    ax.plot(bounds + np.array([-padding, padding]), bounds + np.array([-padding, padding]), color="0.5", linewidth=0.7)
    ax.set_xlim(bounds[0] - padding, bounds[1] + padding)
    ax.set_ylim(bounds[0] - padding, bounds[1] + padding)
    ax.set_xlabel("Current filtered 5-min delta (r)")
    ax.set_ylabel("Context-only all-outside delta (r)")
    ax.set_title("c  Pair-level change", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    return save_figure(fig, out_dir / "context_fairness_current_vs_primary")


def render_pool_audit(sessions: pd.DataFrame, out_dir: Path) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 2.25))
    ax = axes[0]
    for x, column, color, label in [
        (0, "outside_minutes", "#8C8C8C", "Outside"),
        (1, "selected_maze_minutes", "#B7B7B7", "Filtered maze"),
        (2, "annotated_maze_minutes", "#4A5E65", "All maze"),
    ]:
        values = sessions[column].to_numpy(dtype=float)
        jitter = np.random.default_rng(20260820 + x).uniform(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=8, color=color, alpha=0.55)
        ax.plot([x - 0.13, x + 0.13], [np.median(values)] * 2, color="black", linewidth=1.2)
    ax.set_xticks([0, 1, 2], ["Outside", "Filtered\nmaze", "All maze"])
    ax.set_ylabel("Available calcium time (min)")
    ax.set_title("a  Pre-sampling pools", loc="left", fontweight="bold")

    ax = axes[1]
    fractions = sessions["selected_fraction_of_annotated_maze"].to_numpy(dtype=float)
    ax.hist(fractions, bins=np.linspace(0, 1, 16), color="#4A5E65", edgecolor="white")
    ax.set_xlabel("Filtered / annotated maze frames")
    ax.set_ylabel("Sessions")
    ax.set_title("b  Frames removed by behavior filters", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    return save_figure(fig, out_dir / "context_frame_pool_audit")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [path.resolve() for path in args.run_dirs]
    repeats = read_shards(run_dirs, "equal_time_repeat_results.csv")
    feasibility = read_shards(run_dirs, "equal_time_pair_feasibility.csv")
    if repeats.duplicated(PAIR_KEYS + ["repeat"]).any():
        raise ValueError("Duplicate pair/repeat rows across shards")
    if feasibility.duplicated(PAIR_KEYS).any():
        raise ValueError("Duplicate pair feasibility rows across shards")
    expected_counts = repeats.groupby("variant", observed=True).size().reindex(VARIANT_ORDER)
    if not expected_counts.eq(36 * 20).all():
        raise ValueError(f"Expected 720 repeat rows per variant; found {expected_counts.to_dict()}")
    if len(feasibility) != 144 or not feasibility["feasible"].astype(str).str.lower().eq("true").all():
        raise ValueError("Expected 144 feasible pair rows")

    primary = summarize_repeats(repeats)
    accepted = pd.read_csv(args.accepted_pair_summary, dtype={"animal_id": str})
    if len(primary) != 144 or primary["animal_id"].nunique() != 14:
        raise ValueError("Primary summary must contain 144 variant rows from 14 mice")
    primary_counts = primary.groupby("variant", observed=True).size().reindex(VARIANT_ORDER)
    if not primary_counts.eq(36).all():
        raise ValueError(f"Expected 36 primary pairs per variant; found {primary_counts.to_dict()}")

    current_stats = clustered_intercepts(accepted, definition="Current filtered 5 min")
    primary_stats = clustered_intercepts(primary, definition=args.definition_label)
    stats_table = pd.concat([current_stats, primary_stats], ignore_index=True)
    joined, change_tests = definition_change_test(accepted, primary)
    sessions = session_pool_audit(feasibility)
    if not sessions["draw_counts_match_within_session"].all():
        raise ValueError("Maze/outside draw counts do not match within every session")
    if not args.allow_partial_outside and not sessions["all_outside_frames_used"].all():
        raise ValueError("Not every outside-maze frame was used")

    paths = {
        "repeat_results": args.out_dir / "equal_time_repeat_results_combined.csv",
        "pair_feasibility": args.out_dir / "equal_time_pair_feasibility_combined.csv",
        "pair_summary": args.out_dir / f"equal_time_pair_summary_{args.output_tag}.csv",
        "clustered_stats": args.out_dir / "equal_time_clustered_stats_current_vs_primary.csv",
        "pair_change": args.out_dir / "equal_time_pair_change_current_vs_primary.csv",
        "change_tests": args.out_dir / "equal_time_definition_change_clustered_tests.csv",
        "session_pools": args.out_dir / "equal_time_session_frame_pool_audit.csv",
    }
    repeats.to_csv(paths["repeat_results"], index=False)
    feasibility.to_csv(paths["pair_feasibility"], index=False)
    primary.to_csv(paths["pair_summary"], index=False)
    stats_table.to_csv(paths["clustered_stats"], index=False)
    joined.to_csv(paths["pair_change"], index=False)
    change_tests.to_csv(paths["change_tests"], index=False)
    sessions.to_csv(paths["session_pools"], index=False)
    figure_paths = render_comparison(
        accepted,
        primary,
        stats_table,
        joined,
        args.out_dir,
        primary_definition=args.definition_label,
    )
    pool_figure_paths = render_pool_audit(sessions, args.out_dir)

    d1_stats = stats_table.loc[
        stats_table["variant"].eq("d1v1")
        & stats_table["estimand"].isin(["outside", "maze", "maze_minus_outside"])
    ].copy()
    manifest = {
        "analysis_role": "unpromoted context-fairness audit",
        "run_dirs": run_dirs,
        "accepted_pair_summary": args.accepted_pair_summary.resolve(),
        "accepted_pair_summary_sha256": sha256(args.accepted_pair_summary),
        "n_repeat_rows": int(len(repeats)),
        "n_pair_rows": int(len(primary)),
        "n_animals": int(primary["animal_id"].nunique()),
        "n_unique_sessions": int(len(sessions)),
        "sampling_rule": args.sampling_rule_text,
        "maze_frame_rule": "annotated in-maze context only; no position or speed filter",
        "outside_frame_rule": "complement of annotated in-maze intervals",
        "neural_frame_timebase": "original imaging-frame index",
        "technical_repeats": 20,
        "random_seed": 0,
        "statistical_unit": "long-gap session pair; CR1 standard errors clustered by animal",
        "multiple_comparison_family": "BH across four day-combination maze-minus-outside tests",
        "outputs": paths,
        "figures": [*figure_paths, *pool_figure_paths],
    }
    output_files = [*paths.values(), *figure_paths, *pool_figure_paths]
    manifest["output_sha256"] = {str(path): sha256(path) for path in output_files}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")

    report_lines = [
        "# Context-fair equal-duration ensemble-rate audit",
        "",
        "The promoted figure and manuscript text were not changed. " + args.sampling_rule_text + ".",
        "",
        "## D1-D1 estimates",
        "",
        markdown_table(d1_stats),
        "",
        "## Frame-pool checks",
        "",
        f"- {len(sessions)} unique sessions; every in-maze draw exactly matched the corresponding outside-maze draw count.",
        f"- Outside duration range: {sessions['outside_minutes'].min():.2f}-{sessions['outside_minutes'].max():.2f} min.",
        f"- Annotated in-maze duration range: {sessions['annotated_maze_minutes'].min():.2f}-{sessions['annotated_maze_minutes'].max():.2f} min.",
        f"- Filtered rows retained a median {sessions['selected_fraction_of_annotated_maze'].median():.1%} of annotated in-maze frames.",
        "",
        "## Files",
        "",
        *[f"- `{path.name}`" for path in output_files],
        "- `manifest.json`",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
