"""Render ED1 panel-h window comparisons on the matched 153-session cohort.

This is an unpromoted candidate analysis.  It uses the exact behavioral
session registry underlying current ED1 panels b--g and recomputes the existing
continuous body-mass speed estimator for the first 30 min, first 60 min and
all observed cumulative in-maze time.  The accepted ED1 record is read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import analyze_ed01_panel_h_window_sensitivity as base


REPO = Path(__file__).resolve().parents[1]
CURRENT_ED1_RECORD = (
    REPO
    / "docs"
    / "figure_version_records"
    / "extended_data_fig01_2026-08-15_round17_whole_group_first30_shared_scale"
)
MATCHED_COHORT_SOURCE = (
    CURRENT_ED1_RECORD
    / "plot_ready"
    / "ed01_endpoint_axis_session_rows_current153.csv"
)
CURRENT_PANEL_H_SOURCE = (
    CURRENT_ED1_RECORD
    / "plot_ready"
    / "bodymass_speed_current_plus_new12_by_session.csv"
)
DEFAULT_OUT = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed01_panel_h_matched153_window_sensitivity_2026-08-15"
)
EXPECTED_COUNTS = {
    (0, "day1"): 23,
    (0, "day2"): 18,
    (3, "day1"): 17,
    (3, "day2"): 17,
    (6, "day1"): 17,
    (6, "day2"): 17,
    (9, "day1"): 12,
    (9, "day2"): 12,
    (12, "day1"): 10,
    (12, "day2"): 10,
}
WINDOW_ORDER = ("first_30_min", "first_60_min", "whole_session")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--behavior-root", type=Path, default=Path(r"D:\combineddata"))
    parser.add_argument("--reuse-computed", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_matched_registry() -> pd.DataFrame:
    source = pd.read_csv(MATCHED_COHORT_SOURCE, dtype={"animal_id": str})
    required = {"session", "animal_id", "month", "group_key", "source_set", "load_ok"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"matched cohort source lacks columns: {sorted(missing)}")
    registry = source[
        ["session", "animal_id", "month", "group_key", "source_set", "load_ok"]
    ].drop_duplicates("session")
    registry = registry.rename(columns={"session": "session_id"})
    registry["animal_id"] = registry["animal_id"].astype(str)
    registry["month"] = pd.to_numeric(registry["month"], errors="raise").astype(int)
    registry["day"] = np.where(
        registry["group_key"].astype(str).eq("experienced_day2"), "day2", "day1"
    )
    registry["month_label"] = registry["month"].astype(str) + "mo"
    registry["cohort"] = np.where(
        registry["day"].eq("day2"),
        "experienced_day2",
        np.where(registry["month"].eq(0), "naive", "ltm_return"),
    )
    registry["cohort_simple"] = registry["cohort"]
    registry["group"] = (
        registry["cohort"].astype(str)
        + " "
        + registry["month_label"].astype(str)
        + " "
        + registry["day"].astype(str)
    )

    route = pd.read_csv(base.ROUTE_CONTEXT, dtype={"animal": str})[
        ["session_id", "cohort_key", "reward_context"]
    ].drop_duplicates("session_id")
    registry = registry.merge(route, on="session_id", how="left", validate="one_to_one")

    context = pd.read_csv(base.NO_REWARD_CONTEXT, dtype={"animal_id": str})[
        ["session_day1", "cohort_group", "cohort_label", "day1_waterport_status"]
    ].rename(columns={"session_day1": "session_id"})
    registry = registry.merge(
        context, on="session_id", how="left", validate="one_to_one"
    )
    registry["reward_context_defined"] = registry["reward_context"].notna()
    registry["reward_context_note"] = np.where(
        registry["reward_context_defined"],
        registry["reward_context"].astype(str),
        np.where(
            registry["day1_waterport_status"].notna(),
            "day1_" + registry["day1_waterport_status"].astype(str),
            "not_resolved_by_reward_context_sources",
        ),
    )
    registry["registry_source"] = base.rel(MATCHED_COHORT_SOURCE)
    return registry.sort_values(["month", "day", "session_id"]).reset_index(drop=True)


def build_cohort_match_audit(registry: pd.DataFrame) -> pd.DataFrame:
    matched = registry[
        ["session_id", "animal_id", "day", "month", "month_label", "source_set"]
    ].copy()
    matched["in_matched_ed1_registry"] = True

    historical = pd.read_csv(CURRENT_PANEL_H_SOURCE, dtype={"animal_id": str})
    historical = historical.loc[
        historical["position_source"].astype(str).eq(base.POSITION_SOURCE)
        & historical["analysis_mode"].astype(str).eq("explore")
        & pd.to_numeric(historical["scope_min"], errors="coerce").eq(60),
        ["session_id", "animal_id", "day", "month_label", "source_set"],
    ].drop_duplicates("session_id")
    historical["month"] = pd.to_numeric(
        historical["month_label"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="raise",
    ).astype(int)
    historical["in_current_panel_h_cache"] = True

    audit = matched.merge(
        historical,
        on="session_id",
        how="outer",
        suffixes=("_matched", "_panel_h"),
        indicator=True,
        validate="one_to_one",
    )
    audit["in_matched_ed1_registry"] = audit["in_matched_ed1_registry"].eq(True)
    audit["in_current_panel_h_cache"] = audit["in_current_panel_h_cache"].eq(True)
    audit["status"] = audit["_merge"].map(
        {
            "both": "shared",
            "left_only": "missing_from_current_panel_h",
            "right_only": "extra_in_current_panel_h",
        }
    )
    for column in ("animal_id", "day", "month", "month_label", "source_set"):
        audit[column] = audit[f"{column}_matched"].combine_first(
            audit[f"{column}_panel_h"]
        )
    return audit[
        [
            "session_id",
            "animal_id",
            "day",
            "month",
            "month_label",
            "source_set",
            "in_matched_ed1_registry",
            "in_current_panel_h_cache",
            "status",
        ]
    ].sort_values(["month", "day", "session_id"])


def build_inclusion_audit(session_rows: pd.DataFrame) -> pd.DataFrame:
    audit = session_rows[
        [
            "session_id",
            "animal_id",
            "day",
            "month",
            "month_label",
            "analysis_mode",
            "window_id",
            "window_label",
            "eligible_time_sec",
            "valid_time_sec",
            "distance_cm",
            "mode_time_distance_rate_cm_s",
            "reward_context_defined",
            "reward_context_note",
        ]
    ].copy()
    finite_speed = np.isfinite(
        pd.to_numeric(audit["mode_time_distance_rate_cm_s"], errors="coerce")
    )
    audit["included_exact_current_mode_rule"] = finite_speed
    audit["included_reward_context_strict"] = finite_speed & ~(
        audit["analysis_mode"].eq("reward")
        & ~audit["reward_context_defined"].astype(bool)
    )
    audit["exclusion_reason_exact"] = np.where(
        finite_speed, "", "no_finite_mode_speed"
    )
    audit["exclusion_reason_strict"] = np.where(
        ~finite_speed,
        "no_finite_mode_speed",
        np.where(
            audit["analysis_mode"].eq("reward")
            & ~audit["reward_context_defined"].astype(bool),
            "reward_context_not_resolved",
            "",
        ),
    )
    return audit.sort_values(
        ["month", "day", "session_id", "analysis_mode", "window_id"]
    )


def registry_count_table(registry: pd.DataFrame) -> pd.DataFrame:
    counts = (
        registry.groupby(["month", "day"], as_index=False)
        .agg(n_sessions=("session_id", "nunique"), n_animals=("animal_id", "nunique"))
        .sort_values(["month", "day"])
    )
    counts["expected_n"] = [EXPECTED_COUNTS[(int(row.month), str(row.day))] for row in counts.itertuples()]
    counts["count_match"] = counts["n_sessions"].eq(counts["expected_n"])
    return counts


def common_ymax(summary: pd.DataFrame) -> float:
    upper = pd.to_numeric(summary["speed_mean_cm_s"], errors="coerce") + pd.to_numeric(
        summary["speed_sem_cm_s"], errors="coerce"
    ).fillna(0)
    finite = upper[np.isfinite(upper)]
    if finite.empty:
        return 2.0
    return float(max(2.0, 2.0 * math.ceil(float(finite.max()) / 2.0)))


def sample_note() -> str:
    return (
        "Registry sessions at 0, 3, 6, 9, 12 months:\n"
        "Day 1: 23, 17, 17, 12, 10; Day 2: 18, 17, 17, 12, 10"
    )


def render_window_grid(
    group_summary: pd.DataFrame,
    *,
    windows: tuple[str, ...],
    variant: str,
    stem: Path,
) -> tuple[list[Path], dict[str, Any]]:
    summary = group_summary.loc[
        group_summary["summary_variant"].eq(variant)
        & group_summary["window_id"].isin(windows)
    ].copy()
    rows = len(windows)
    figure_height = 2.75 if rows == 1 else (5.0 if rows == 2 else 7.0)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(5.2, figure_height),
        sharex=rows > 1,
        sharey=True,
        squeeze=False,
    )
    ymax = common_ymax(summary)
    yticks = np.arange(0, ymax + 0.01, 2.0)
    for row_index, window_id in enumerate(windows):
        window_summary = summary.loc[summary["window_id"].eq(window_id)]
        for column_index, mode in enumerate(base.MODES):
            axis = axes[row_index, column_index]
            base.draw_speed_axis(
                axis,
                window_summary,
                mode=mode,
                show_ylabel=column_index == 0,
                show_xlabel=row_index == rows - 1,
                legend=row_index == 0 and column_index == 1,
            )
            axis.set_ylim(0, ymax)
            axis.set_yticks(yticks)
            if column_index == 0:
                axis.set_ylabel("Body mass speed (cm s$^{-1}$)")
            if row_index > 0:
                axis.set_title("")
        axes[row_index, 0].text(
            0.03,
            0.96,
            str(window_summary["window_label"].iloc[0]),
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.0,
            },
        )
    title = "Body mass speed in the matched ED1 behavioral cohort"
    if variant == "reward_context_strict":
        title += " (reward-context strict)"
    fig.suptitle(title, y=0.995, fontsize=8.1)
    fig.text(0.55, 0.015, sample_note(), ha="center", va="bottom", fontsize=5.4)
    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        bottom=0.23 if rows == 1 else (0.14 if rows == 2 else 0.10),
        top=0.80 if rows == 1 else 0.91,
        wspace=0.16,
        hspace=0.30,
    )
    paths = base.save_figure(fig, stem)
    ylimits = [list(axis.get_ylim()) for axis in axes.flat]
    plt.close(fig)
    return paths, {"ymax": ymax, "yticks": yticks.tolist(), "ylimits": ylimits}


def validate(
    *,
    out_dir: Path,
    registry: pd.DataFrame,
    counts: pd.DataFrame,
    cohort_audit: pd.DataFrame,
    session_rows: pd.DataFrame,
    bin_rows: pd.DataFrame,
    source_qc: pd.DataFrame,
    failures: pd.DataFrame,
    group_summary: pd.DataFrame,
    figure_paths: list[Path],
    figure_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    check("matched_registry_has_153_sessions", len(registry) == 153, len(registry))
    check(
        "matched_registry_group_counts",
        len(counts) == len(EXPECTED_COUNTS) and counts["count_match"].all(),
        counts.to_dict(orient="records"),
    )
    check(
        "matched_registry_source_rows_all_load_ok",
        registry["load_ok"].astype(str).str.lower().eq("true").all(),
        registry["load_ok"].value_counts(dropna=False).to_dict(),
    )
    check("raw_recompute_has_no_failures", failures.empty, len(failures))
    check("one_source_qc_row_per_session", len(source_qc) == 153, len(source_qc))
    check(
        "complete_session_mode_window_grid",
        len(session_rows) == 153 * len(base.MODES) * len(base.WINDOWS),
        len(session_rows),
    )
    check("bin_level_conversion_exported", not bin_rows.empty, len(bin_rows))
    first60 = session_rows.loc[session_rows["window_id"].eq("first_60_min")]
    first60_finite = np.isfinite(
        pd.to_numeric(first60["mode_time_distance_rate_cm_s"], errors="coerce")
    )
    check(
        "all_matched_sessions_have_first60_mode_support",
        first60_finite.all() and len(first60) == 153 * len(base.MODES),
        {"finite": int(first60_finite.sum()), "rows": int(len(first60))},
    )
    exact = group_summary.loc[
        group_summary["summary_variant"].eq("exact_current_mode_rule")
    ]
    check("complete_exact_group_grid", len(exact) == 60, len(exact))
    twelve = counts.loc[counts["month"].eq(12)].set_index("day")["n_sessions"].to_dict()
    check("twelve_month_registry_is_10_plus_10", twelve == {"day1": 10, "day2": 10}, twelve)
    twelve_audit = cohort_audit.loc[pd.to_numeric(cohort_audit["month"], errors="coerce").eq(12)]
    check(
        "historical_panel_h_12mo_gap_is_12_recordings",
        int(twelve_audit["in_matched_ed1_registry"].sum()) == 20
        and int(twelve_audit["in_current_panel_h_cache"].sum()) == 8
        and int(twelve_audit["status"].eq("missing_from_current_panel_h").sum()) == 12,
        twelve_audit["status"].value_counts().to_dict(),
    )
    check(
        "all_figure_exports_exist",
        all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
        [base.rel(path) for path in figure_paths],
    )
    check(
        "every_grid_uses_one_common_y_scale",
        all(
            len({tuple(limit) for limit in metadata["ylimits"]}) == 1
            for metadata in figure_metadata.values()
        ),
        figure_metadata,
    )
    png_paths = [path for path in figure_paths if path.suffix.lower() == ".png"]
    png_sizes = {path.name: list(Image.open(path).size) for path in png_paths}
    check(
        "png_previews_have_publication_resolution",
        all(min(size) >= 1000 for size in png_sizes.values()),
        png_sizes,
    )
    tiff_info = {}
    for path in [path for path in figure_paths if path.suffix.lower() == ".tiff"]:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            tiff_info[path.name] = {
                "size": list(image.size),
                "dpi": [float(value) for value in dpi] if dpi else None,
                "compression": image.info.get("compression"),
            }
    check(
        "tiff_exports_are_600dpi_lzw",
        all(
            info["dpi"] is not None
            and min(info["dpi"]) >= 599
            and str(info["compression"]).lower() == "tiff_lzw"
            for info in tiff_info.values()
        ),
        tiff_info,
    )
    pdf_images = {}
    for path in [path for path in figure_paths if path.suffix.lower() == ".pdf"]:
        with fitz.open(path) as document:
            pdf_images[path.name] = sum(len(page.get_images(full=True)) for page in document)
    check(
        "pdf_exports_contain_no_raster_images",
        all(value == 0 for value in pdf_images.values()),
        pdf_images,
    )
    svg_info = {}
    for path in [path for path in figure_paths if path.suffix.lower() == ".svg"]:
        text = path.read_text(encoding="utf-8")
        svg_info[path.name] = {"editable_text": "<text" in text, "has_image": "<image" in text}
    check(
        "svg_exports_have_editable_text_and_no_images",
        all(row["editable_text"] and not row["has_image"] for row in svg_info.values()),
        svg_info,
    )
    validation = {
        "pass": all(row["pass"] for row in checks.values()),
        "passed": int(sum(row["pass"] for row in checks.values())),
        "total": len(checks),
        "checks": checks,
    }
    (out_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return validation


def write_report(
    *,
    out_dir: Path,
    counts: pd.DataFrame,
    cohort_audit: pd.DataFrame,
    group_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    source_qc: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    lines = [
        "# ED1 panel-h matched 153-session window sensitivity",
        "",
        "## Result",
        "",
        "The historical panel-h cache was not cohort-matched to the behavioral panels. This candidate recomputes the same body-mass speed estimator on the exact 153-session ED1 behavioral registry. At 12 months it includes 10 Day 1 and 10 Day 2 recordings, rather than 4 and 4.",
        "",
        "## Cohort accounting",
        "",
        "| month | day | matched sessions | expected sessions |",
        "|---:|---|---:|---:|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {int(row.month)} | {row.day} | {int(row.n_sessions)} | {int(row.expected_n)} |")
    twelve = cohort_audit.loc[pd.to_numeric(cohort_audit["month"], errors="coerce").eq(12)]
    missing = twelve.loc[twelve["status"].eq("missing_from_current_panel_h"), "session_id"].tolist()
    lines.extend(
        [
            "",
            f"The current panel-h cache contains {int(twelve['in_current_panel_h_cache'].sum())} of the 20 twelve-month recordings. The matched candidate restores: {', '.join(missing)}.",
            "",
            "## Estimand",
            "",
            "The statistical unit is one recording session. Continuous normalized body-center coordinates are median-smoothed within continuous trajectory runs over five frames. Only adjacent frames within the same in-maze bout contribute. Segments above 80 cm/s are excluded. Speed is total retained distance divided by eligible operational mode time. Windows use cumulative in-maze time; whole session means every observed in-maze bout and is not a fixed-duration cap. Group symbols and error bars are means +/- SEM across finite session-level estimates.",
            "",
            "## Plot-ready results",
            "",
            "| mode | window | day | month | mean +/- SEM (cm/s) | finite n / registry n |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    exact = group_summary.loc[
        group_summary["summary_variant"].eq("exact_current_mode_rule")
    ].sort_values(["analysis_mode", "window_order", "day", "month"])
    for row in exact.itertuples(index=False):
        lines.append(
            f"| {row.analysis_mode} | {row.window_label} | {row.day} | {int(row.month)} | "
            f"{row.speed_mean_cm_s:.3f} +/- {row.speed_sem_cm_s:.3f} | "
            f"{int(row.n_sessions)} / {int(row.n_sessions_registry)} |"
        )
    exact_comparison = comparison.loc[
        comparison["summary_variant"].eq("exact_current_mode_rule")
    ]
    delta = pd.to_numeric(
        exact_comparison["whole_minus_first30_speed_cm_s"], errors="coerce"
    )
    lines.extend(
        [
            "",
            "## Conversion and validation audit",
            "",
            f"- Raw and trajectory inputs loaded for all 153 sessions; cumulative in-maze duration ranged from {source_qc['whole_session_maze_time_min'].min():.1f} to {source_qc['whole_session_maze_time_min'].max():.1f} min.",
            f"- Whole-minus-first-30 group differences ranged from {delta.min():+.3f} to {delta.max():+.3f} cm/s across modes, days and months.",
            "- `cohort_match_audit.csv` exposes every shared, omitted and extra historical panel-h session.",
            "- `summary_inclusion_audit.csv` exposes finite-mode and reward-context inclusion masks before aggregation.",
            "- `speed_by_5min_bin.csv` and `speed_by_session_window.csv` expose the bin-to-session transformation before group means.",
            f"- Validation passed {validation['passed']}/{validation['total']} checks.",
            "- This is an unpromoted candidate. The accepted ED1 record, pointers and manuscript captions were not modified.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_existing(out_dir: Path) -> int:
    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    validation = json.loads((out_dir / "VALIDATION.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["output_files"]:
        path = REPO / item["path"]
        if not path.is_file() or base.sha256(path) != item["sha256"]:
            failures.append(item["path"])
    print(json.dumps({"validation": validation, "hash_failures": failures}, indent=2))
    return 0 if validation.get("pass") and not failures else 1


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if args.validate_only:
        return validate_existing(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.configure_style()

    registry = load_matched_registry()
    cohort_audit = build_cohort_match_audit(registry)
    counts = registry_count_table(registry)
    if args.reuse_computed:
        session_rows = pd.read_csv(
            out_dir / "speed_by_session_window.csv", dtype={"animal_id": str}
        )
        bin_rows = pd.read_csv(out_dir / "speed_by_5min_bin.csv", dtype={"animal_id": str})
        source_qc = pd.read_csv(out_dir / "session_source_qc.csv", dtype={"animal_id": str})
        try:
            failures = pd.read_csv(out_dir / "failures.csv")
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()
    else:
        session_rows, bin_rows, source_qc, failures = base.recompute(
            registry, behavior_root=args.behavior_root.resolve()
        )

    exact = base.summarize_groups(session_rows, strict_reward_context=False)
    strict = base.summarize_groups(session_rows, strict_reward_context=True)
    group_summary = pd.concat([exact, strict], ignore_index=True)
    comparison = base.build_window_comparison(group_summary)
    inclusion = build_inclusion_audit(session_rows)

    registry.to_csv(out_dir / "session_registry.csv", index=False)
    counts.to_csv(out_dir / "registry_group_counts.csv", index=False)
    cohort_audit.to_csv(out_dir / "cohort_match_audit.csv", index=False)
    source_qc.to_csv(out_dir / "session_source_qc.csv", index=False)
    failures.to_csv(out_dir / "failures.csv", index=False)
    session_rows.to_csv(out_dir / "speed_by_session_window.csv", index=False)
    bin_rows.to_csv(out_dir / "speed_by_5min_bin.csv", index=False)
    inclusion.to_csv(out_dir / "summary_inclusion_audit.csv", index=False)
    group_summary.to_csv(out_dir / "group_summary.csv", index=False)
    comparison.to_csv(out_dir / "window_comparison.csv", index=False)

    figure_paths: list[Path] = []
    figure_metadata: dict[str, dict[str, Any]] = {}
    figure_specs = [
        ("matched153_first_30_min", ("first_30_min",), "exact_current_mode_rule"),
        ("matched153_whole_session", ("whole_session",), "exact_current_mode_rule"),
        (
            "matched153_first30_vs_whole",
            ("first_30_min", "whole_session"),
            "exact_current_mode_rule",
        ),
        (
            "matched153_all_windows",
            WINDOW_ORDER,
            "exact_current_mode_rule",
        ),
        (
            "matched153_first30_vs_whole_reward_context_strict",
            ("first_30_min", "whole_session"),
            "reward_context_strict",
        ),
    ]
    for name, windows, variant in figure_specs:
        paths, metadata = render_window_grid(
            group_summary,
            windows=windows,
            variant=variant,
            stem=out_dir / f"ed01_panel_h_{name}",
        )
        figure_paths.extend(paths)
        figure_metadata[name] = metadata

    validation = validate(
        out_dir=out_dir,
        registry=registry,
        counts=counts,
        cohort_audit=cohort_audit,
        session_rows=session_rows,
        bin_rows=bin_rows,
        source_qc=source_qc,
        failures=failures,
        group_summary=group_summary,
        figure_paths=figure_paths,
        figure_metadata=figure_metadata,
    )
    write_report(
        out_dir=out_dir,
        counts=counts,
        cohort_audit=cohort_audit,
        group_summary=group_summary,
        comparison=comparison,
        source_qc=source_qc,
        validation=validation,
    )

    output_paths = [
        path for path in sorted(out_dir.iterdir()) if path.is_file() and path.name != "MANIFEST.json"
    ]
    manifest = {
        "analysis": "ED1 panel-h matched 153-session window sensitivity",
        "generated": "2026-08-15",
        "status": "unpromoted_candidate",
        "backend": "Python/matplotlib only",
        "accepted_record_unchanged": True,
        "figure_contract": {
            "core_conclusion": "compare first-30-min, first-60-min and whole-session body-mass speed on the exact ED1 behavioral cohort",
            "statistical_unit": "recording session",
            "cohort": "153 sessions from the current ED1 b-g source registry",
            "representation": "continuous normalized body-center position converted to centimeters",
            "timebase": "cumulative in-maze time from half-open trajectory-bout frame intervals",
            "estimator": "total retained adjacent-frame distance divided by eligible operational mode time",
            "windows": ["first 30 min", "first 60 min", "all observed in-maze time"],
            "error_bars": "SEM across finite session-level estimates",
            "inference": "descriptive; no new significance tests",
            "scale_policy": "one common y limit and tick set across directly comparable axes",
        },
        "fixed_settings": {
            "position_source": base.POSITION_SOURCE,
            "modes": list(base.MODES),
            "median_window_frames": base.MEDIAN_WINDOW,
            "max_segment_speed_cm_s": base.MAX_SEGMENT_SPEED_CM_S,
            "bin_size_min": base.BIN_SIZE_MIN,
            "maze_id": base.MAZE_ID,
            "fps": base.FPS,
        },
        "source_files": [
            {"path": base.rel(MATCHED_COHORT_SOURCE), "sha256": base.sha256(MATCHED_COHORT_SOURCE)},
            {"path": base.rel(CURRENT_PANEL_H_SOURCE), "sha256": base.sha256(CURRENT_PANEL_H_SOURCE)},
            {"path": base.rel(base.ROUTE_CONTEXT), "sha256": base.sha256(base.ROUTE_CONTEXT)},
            {"path": base.rel(base.NO_REWARD_CONTEXT), "sha256": base.sha256(base.NO_REWARD_CONTEXT)},
        ],
        "figure_metadata": figure_metadata,
        "validation": validation,
        "output_files": [
            {"path": base.rel(path), "sha256": base.sha256(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": base.rel(out_dir), "validation": validation}, indent=2))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
