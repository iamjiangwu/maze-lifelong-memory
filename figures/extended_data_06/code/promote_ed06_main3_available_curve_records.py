"""Promote ED6 and Figure 3 caption-analysis successor records.

The plotted curve sources already use the maintained current59 authority, and
neither figure displays the newly reported curve statistics. Therefore the
artwork is intentionally byte-identical to the accepted parents. The successor
records replace stale caption-facing support/statistics metadata, freeze the
all-available-curve audit, and close append-only record hashes.

Run ``--create`` first. Update the two current-pointer documents only after the
successors validate, then run ``--archive`` to move both parents intact into the
archive and update its tree ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs" / "figure_version_records"
ANALYSIS = REPO / "docs" / "ed06_figure3_available_curve_support_2026-08-16"
CAPTION = (
    REPO
    / "docs"
    / "manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)

ED6_PARENT = "extended_data_fig06_2026-08-16_round6_upright_i"
ED6_TARGET = "extended_data_fig06_2026-08-16_round7_available_curve_all59"
MAIN3_PARENT = "main_figure3_2026-08-14_round8_bh4_correlations"
MAIN3_TARGET = "main_figure3_2026-08-16_round9_available_window_slopes"

ARTWORK = {
    "ed6": (
        "ed06_version_record.svg",
        "ed06_version_record.pdf",
        "ed06_version_record_300dpi.png",
        "ed06_version_record_600dpi.png",
        "ed06_version_record_600dpi.tiff",
    ),
    "main3": (
        "figure3_version_record.html",
        "figure3_version_record.svg",
        "figure3_version_record.pdf",
        "figure3_version_record_300dpi.png",
        "figure3_version_record_600dpi.png",
        "figure3_version_record_600dpi.tiff",
    ),
}

ANALYSIS_FILES = (
    "REPORT.md",
    "cohort_boundary_840_841.csv",
    "current59_recording_cycle_roster.csv",
    "ed06_available_curve_clustered_stats.csv",
    "ed06_pair_metric_available_curve_summaries.csv",
    "figure3_30min_pair_slopes_current59.csv",
    "figure3_30min_slope_validation.csv",
    "manifest.json",
    "validation.json",
)

sys.path.insert(0, str((REPO / "scripts").resolve()))
from promote_all_figure_refinement_round2_records import (  # noqa: E402
    Spec,
    archive_parent,
    sha256,
    update_archive_manifest,
    verify_hashes,
    write_hashes,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_caption_section(heading: str, next_heading: str) -> str:
    text = CAPTION.read_text(encoding="utf-8")
    start = text.index(heading)
    end = text.index(next_heading, start)
    return text[start:end].rstrip() + "\n"


def _copy_analysis_bundle(stage: Path) -> Path:
    destination = stage / "source_data" / "available_curve_support_2026_08_16"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ANALYSIS_FILES:
        source = ANALYSIS / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    caption_dir = stage / "source_data" / "manuscript_caption"
    caption_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CAPTION, caption_dir / CAPTION.name)
    shutil.copy2(
        REPO / "scripts" / "analyze_ed06_figure3_available_curve_support.py",
        stage / "source_code" / "analyze_ed06_figure3_available_curve_support.py",
    )
    shutil.copy2(Path(__file__), stage / "source_code" / Path(__file__).name)
    return destination


def _star(adjusted_p: float) -> str:
    if adjusted_p <= 0.001:
        return "***"
    if adjusted_p <= 0.01:
        return "**"
    if adjusted_p <= 0.05:
        return "*"
    return "n.s."


def _ed6_caption_statistics(parent: Path) -> pd.DataFrame:
    inherited = pd.read_csv(parent / "ED06_DISPLAYED_STATISTICS.csv")
    fixed = inherited[inherited["panel"].isin(["c", "e", "g"])].copy()
    fixed_output = pd.DataFrame(
        {
            "panel": fixed["panel"],
            "metric": fixed["metric"],
            "column": fixed["column"],
            "estimand": fixed["estimand"],
            "estimate": fixed["estimate"],
            "unit": fixed["unit"],
            "p_two_sided": pd.NA,
            "p_bh_two_sided": fixed["q_bh18_two_sided"],
            "stars": fixed["stars"],
            "n_pair_rows": fixed["n_pair_rows"],
            "n_animal_clusters": fixed["n_animal_clusters"],
            "mean_day_i_within_curve_sd": pd.NA,
            "mean_day_i1_within_curve_sd": pd.NA,
            "correction_family": "BH18 fixed-reference tests",
        }
    )

    stats = pd.read_csv(ANALYSIS / "ed06_available_curve_clustered_stats.csv")
    panel_map = {
        "place_map_correlation": "d",
        "pv_correlation": "f",
        "ensemble_rate_correlation": "h",
    }
    column_map = {
        "day_i1_minus_day_i_within_curve_sd": "Day i+1 minus Day i",
        "day_i_linear_slope": "Day i",
        "day_i1_linear_slope": "Day i+1",
    }
    unit_map = {
        "day_i1_minus_day_i_within_curve_sd": "r",
        "day_i_linear_slope": "r per hour",
        "day_i1_linear_slope": "r per hour",
    }
    current_rows: list[dict[str, Any]] = []
    for row in stats.itertuples(index=False):
        adjusted = float(row.p_bh3_two_sided)
        current_rows.append(
            {
                "panel": panel_map[row.metric],
                "metric": row.metric,
                "column": column_map[row.estimand],
                "estimand": row.estimand,
                "estimate": float(row.estimate),
                "unit": unit_map[row.estimand],
                "p_two_sided": float(row.p_two_sided),
                "p_bh_two_sided": adjusted,
                "stars": _star(adjusted),
                "n_pair_rows": int(row.n_recording_cycles),
                "n_animal_clusters": int(row.n_animals),
                "mean_day_i_within_curve_sd": getattr(
                    row, "mean_day_i_within_curve_sd", None
                ),
                "mean_day_i1_within_curve_sd": getattr(
                    row, "mean_day_i1_within_curve_sd", None
                ),
                "correction_family": str(row.correction_family),
            }
        )
    return pd.DataFrame(fixed_output.to_dict(orient="records") + current_rows)


def _update_ed6(stage: Path, parent: Path, bundle: Path) -> None:
    caption_stats = _ed6_caption_statistics(parent)
    caption_stats.to_csv(stage / "ED06_DISPLAYED_STATISTICS.csv", index=False)

    validation = pd.read_csv(stage / "ED06_VALIDATION.csv")
    validation = validation[
        ~validation["check"].isin(
            ["pivot_scope_30min_56_pairs_22_mice", "pivot_bh9_family_per_day"]
        )
    ]
    validation = pd.concat(
        [
            validation,
            pd.DataFrame(
                [
                    {
                        "check": "pivot_scope_30min_all59_available_curves",
                        "status": "PASS",
                        "detail": "59 of 59 cycles retained for each metric; shortest common curve has 24 pivots from 30 to 53 min",
                    },
                    {
                        "check": "pivot_caption_inference_bh3_by_estimand",
                        "status": "PASS",
                        "detail": "BH3 separately across metrics for within-curve s.d. contrast, Day i slope and Day i+1 slope",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    validation.to_csv(stage / "ED06_VALIDATION.csv", index=False)

    manifest_path = stage / "ED06_ASSEMBLY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["round"] = 7
    manifest["caption_analysis"] = {
        "support_rule": "all exactly joint valid Day i and Day i+1 30-min pivots within each recording cycle; minimum 3 pivots; no minimum span",
        "n_recording_cycles": 59,
        "n_animals": 22,
        "shortest_curve": "24 pivots from 30 to 53 min",
        "pair_summaries": "source_data/available_curve_support_2026_08_16/ed06_pair_metric_available_curve_summaries.csv",
        "statistics": "source_data/available_curve_support_2026_08_16/ed06_available_curve_clustered_stats.csv",
        "caption": f"source_data/manuscript_caption/{CAPTION.name}",
    }
    manifest["sources"]["available_curve_pair_summaries"] = {
        "path": "source_data/available_curve_support_2026_08_16/ed06_pair_metric_available_curve_summaries.csv",
        "sha256": sha256(bundle / "ed06_pair_metric_available_curve_summaries.csv"),
        "rows": 177,
    }
    manifest["sources"]["available_curve_clustered_stats"] = {
        "path": "source_data/available_curve_support_2026_08_16/ed06_available_curve_clustered_stats.csv",
        "sha256": sha256(bundle / "ed06_available_curve_clustered_stats.csv"),
        "rows": 9,
    }
    manifest["displayed_statistics"] = json.loads(
        caption_stats[caption_stats["panel"].isin(["c", "e", "g"])].to_json(
            orient="records"
        )
    )
    manifest["caption_statistics"] = json.loads(
        caption_stats.to_json(orient="records")
    )
    manifest["validation"] = [
        item
        for item in manifest.get("validation", [])
        if item.get("check")
        not in {"pivot_scope_30min_56_pairs_22_mice", "pivot_bh9_family_per_day"}
    ]
    manifest["validation"].extend(
        [
            {
                "check": "pivot_scope_30min_all59_available_curves",
                "status": "PASS",
                "detail": "59 of 59 cycles retained for each metric; shortest common curve has 24 pivots from 30 to 53 min",
            },
            {
                "check": "pivot_caption_inference_bh3_by_estimand",
                "status": "PASS",
                "detail": "BH3 separately across metrics for within-curve s.d. contrast, Day i slope and Day i+1 slope",
            },
        ]
    )
    _write_json(manifest_path, manifest)

    stage.joinpath("CAPTION_EXCERPT.tex").write_text(
        _extract_caption_section(
            r"\textbf{Extended Data Figure 6.",
            r"\textbf{Extended Data Figure 7.",
        ),
        encoding="utf-8",
    )
    stage.joinpath("CAPTION_NOTES.md").write_text(
        """# Extended Data Figure 6 — caption notes (round 7)

The artwork is unchanged because panels d, f and h already draw the maintained
current59 group curves and carry no numerical statistics or significance marks.
The caption-facing inference now retains all 59 recording cycles from 22 animals.

For each cycle and metric, Day i and Day i+1 are restricted to every exactly
shared valid 30-min pivot. Within-curve s.d. and a linear slope are calculated
separately for each day before pooling cycle-level summaries with CR1 standard
errors clustered by animal. BH correction is applied separately across the three
metrics for the s.d. contrast, Day i slopes and Day i+1 slopes. No minimum curve
span is imposed; the shortest retained curves contain 24 pivots from 30 to 53 min.

Day i+1 has lower whole-curve variability for place maps and population vectors,
but not ensemble rate. None of the three Day i+1 slopes is distinguishable from
zero after correction. The x axis is cumulative elapsed time in maze, not running
time, and each pivot side still requires at least 300 s of included neural data.

Authoritative values are in `ED06_DISPLAYED_STATISTICS.csv`; transformed pair
summaries, the exact current59 roster and support ranges are frozen under
`source_data/available_curve_support_2026_08_16/`. The exact manuscript wording
is in `CAPTION_EXCERPT.tex`.
""",
        encoding="utf-8",
    )


def _update_main3(stage: Path, bundle: Path) -> None:
    stats = pd.read_csv(bundle / "figure3_30min_slope_validation.csv")
    manifest_path = stage / "figure3_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["caption_curve_statistics"] = json.loads(stats.to_json(orient="records"))
    manifest["caption_file"] = f"source_data/manuscript_caption/{CAPTION.name}"
    manifest["sources"].append(
        {
            "panel": "figure3_day2_template_curve_slopes",
            "panels": ["e", "h", "k"],
            "source_path": "source_data/available_curve_support_2026_08_16/figure3_30min_pair_slopes_current59.csv",
            "stats_path": "source_data/available_curve_support_2026_08_16/figure3_30min_slope_validation.csv",
            "filters": {
                "metrics": [
                    "place_map_correlation",
                    "pv_correlation",
                    "ensemble_rate_correlation",
                ],
                "window_type": "maze_time",
                "window_size": 30.0,
            },
            "statistical_unit": "one slope per recording cycle and metric",
            "support_rule": "all finite available Day i windows within each of the frozen current59 recording cycles",
            "inference": "intercept-only OLS on pair-level slopes with CR1 animal-clustered covariance, Student t(G-1), BH18",
        }
    )
    _write_json(manifest_path, manifest)
    stage.joinpath("CAPTION_EXCERPT.tex").write_text(
        _extract_caption_section(r"\textbf{Figure 3.", r"\textbf{Figure 4."),
        encoding="utf-8",
    )
    stage.joinpath("CAPTION_UPDATE.md").write_text(
        """# Figure 3 curve-support caption update

Panels e, h and k already use the complete maintained current59 fixed-reference
curve authority, so the artwork is unchanged. All 59 recording cycles from 22
animals contribute one pair-level 30-min maze-time slope per metric, calculated
from every finite available Day i window. Day i+1 supplies the complete template
and does not determine the moving Day i endpoint.

Mean slopes are 0.0867, 0.0907 and 0.0648 r per hour for place maps, population
vectors and ensemble rates. BH18-adjusted p values are 2.5e-12, 2.6e-11 and
3.1e-6. Exact pair-level inputs and saved-versus-recomputed validation are frozen
under `source_data/available_curve_support_2026_08_16/`; exact manuscript wording
is in `CAPTION_EXCERPT.tex`.
""",
        encoding="utf-8",
    )


def _validate_artwork(parent: Path, stage: Path, names: tuple[str, ...]) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for name in names:
        source = parent / name
        target = stage / name
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(name)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Artwork changed unexpectedly: {name}")
        hashes[name] = {"sha256": target_hash, "bytes": target.stat().st_size}

    pdf_name = next(name for name in names if name.endswith(".pdf"))
    png_name = next(name for name in names if name.endswith("_600dpi.png"))
    with fitz.open(stage / pdf_name) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf_name}")
        page_rect = document[0].rect
    with Image.open(stage / png_name) as image:
        dpi = image.info.get("dpi", (0, 0))
        pixel_size = list(image.size)
    return {
        "files": hashes,
        "pdf_page_count": 1,
        "pdf_size_pt": [page_rect.width, page_rect.height],
        "png600_pixel_size": pixel_size,
        "png600_declared_dpi": list(dpi),
    }


def _create_one(key: str, parent_name: str, target_name: str) -> Path:
    parent = RECORDS / parent_name
    target = RECORDS / target_name
    stage = RECORDS / f".{target_name}.staging"
    if target.exists() or stage.exists():
        raise FileExistsError(target if target.exists() else stage)
    parent_errors = verify_hashes(parent)
    if parent_errors:
        raise RuntimeError(f"Parent record is not closed: {parent_errors[:10]}")
    analysis_validation = json.loads((ANALYSIS / "validation.json").read_text(encoding="utf-8"))
    if analysis_validation.get("status") != "PASS":
        raise RuntimeError("Available-curve analysis validation is not PASS")

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        shutil.copytree(parent, stage)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        bundle = _copy_analysis_bundle(stage)
        if key == "ed6":
            _update_ed6(stage, parent, bundle)
        elif key == "main3":
            _update_main3(stage, bundle)
        else:
            raise ValueError(key)

        art = _validate_artwork(parent, stage, ARTWORK[key])
        promotion = {
            "schema": "available-curve-caption-record-promotion/v1",
            "status": "PASS",
            "record": target_name,
            "parent": parent_name,
            "created_utc": timestamp,
            "artwork_policy": "byte-identical; plotted source rows and visual statistics are unchanged",
            "analysis": {
                "path": "source_data/available_curve_support_2026_08_16",
                "status": "PASS",
                "n_recording_cycles": 59,
                "n_animals": 22,
                "minimum_joint_pivots": 24,
                "shortest_joint_span_min": 23.0,
            },
            "caption": {
                "path": f"source_data/manuscript_caption/{CAPTION.name}",
                "sha256": sha256(CAPTION),
            },
            "artwork": art,
        }
        _write_json(stage / "AVAILABLE_CURVE_SUPPORT_PROMOTION.json", promotion)
        stage.joinpath("MANIFEST.md").write_text(
            f"# {target_name}\n\n"
            f"This append-only record supersedes `{parent_name}`. It freezes the corrected "
            "all-available-curve analysis and manuscript caption provenance for 59 recording "
            "cycles from 22 animals. The accepted artwork is byte-identical to the parent "
            "because the plotted current59 curves and all visible annotations are unchanged; "
            "only caption-facing inference, cohort/support documentation and record metadata "
            "change. See `AVAILABLE_CURVE_SUPPORT_PROMOTION.json`.\n",
            encoding="utf-8",
        )
        stage.joinpath("REBUILD_README.md").write_text(
            f"# Rebuild {target_name}\n\n"
            "The visual exports are inherited byte-for-byte from the accepted parent because "
            "no plotted data or visible annotation changed. Recompute caption-facing statistics "
            "with `source_code/analyze_ed06_figure3_available_curve_support.py`, then recreate "
            "this immutable successor with `source_code/promote_ed06_main3_available_curve_records.py`.\n",
            encoding="utf-8",
        )
        checks = {
            "append_only_successor": True,
            "parent_record_hash_closure": True,
            "analysis_validation_pass": True,
            "current59_cycles": analysis_validation["current59_unique_cycles"] == 59,
            "current59_animals": analysis_validation["current59_unique_animals"] == 22,
            "all_ed6_metrics_retain_59": all(
                value == 59 for value in analysis_validation["ed06_cycles_by_metric"].values()
            ),
            "all_figure3_metrics_retain_59": all(
                value == 59
                for value in analysis_validation["figure3_cycles_by_metric"].values()
            ),
            "caption_parity_checks_pass": all(
                analysis_validation["caption_checks"].values()
            ),
            "artwork_byte_identical_to_parent": True,
            "single_page_pdf": art["pdf_page_count"] == 1,
            "png600_declared_at_least_590dpi": min(art["png600_declared_dpi"]) >= 590,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Successor validation failed: {checks}")
        _write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "available-curve-caption-record-validation/v1",
                "record": target_name,
                "parent": parent_name,
                "status": "PASS",
                "validated_utc": timestamp,
                "checks": checks,
            },
        )
        stage.joinpath("PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={target_name}\ncreated_utc={timestamp}\n",
            encoding="utf-8",
        )
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Record hash closure failed: {errors[:10]}")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def create() -> None:
    created = [
        _create_one("ed6", ED6_PARENT, ED6_TARGET),
        _create_one("main3", MAIN3_PARENT, MAIN3_TARGET),
    ]
    print(
        json.dumps(
            {
                "status": "PASS",
                "created": [path.relative_to(REPO).as_posix() for path in created],
            },
            indent=2,
        )
    )


def archive() -> None:
    specs = [
        Spec(
            "ed6_available_curve",
            ED6_PARENT,
            ED6_TARGET,
            "",
            "",
            "ed06_version_record",
            ("all-available-curve caption analysis",),
        ),
        Spec(
            "main3_available_curve",
            MAIN3_PARENT,
            MAIN3_TARGET,
            "",
            "",
            "figure3_version_record",
            ("all-available-window curve slopes",),
        ),
    ]
    rows = [archive_parent(spec) for spec in specs]
    update_archive_manifest(rows)
    print(json.dumps({"status": "PASS", "archived": rows}, indent=2))


def verify() -> None:
    results: dict[str, Any] = {}
    for key, target_name in (("ed6", ED6_TARGET), ("main3", MAIN3_TARGET)):
        target = RECORDS / target_name
        errors = verify_hashes(target)
        validation = json.loads((target / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
        results[key] = {
            "record": target_name,
            "hash_closure": not errors,
            "record_validation": validation.get("status"),
            "artwork": {
                name: sha256(target / name) for name in ARTWORK[key]
            },
        }
        if errors or validation.get("status") != "PASS":
            raise RuntimeError(results[key])
    print(json.dumps({"status": "PASS", "records": results}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create", action="store_true")
    actions.add_argument("--archive", action="store_true")
    actions.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.create:
        create()
    elif args.archive:
        archive()
    else:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
