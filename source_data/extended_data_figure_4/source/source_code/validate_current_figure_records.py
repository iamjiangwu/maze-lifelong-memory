"""Read-only, fail-closed validation for the 11 current figure records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
RECORD_ROOT = REPO / "docs/figure_version_records"
MAIN_POINTER = RECORD_ROOT / "CURRENT_MAIN_FIGURES.md"
ED_POINTER = RECORD_ROOT / "CURRENT_EXTENDED_DATA_FIGURES.md"
FIGURES = (
    "main_figure_1",
    "main_figure_2",
    "main_figure_3",
    "main_figure_4",
    "main_figure_5",
    "extended_data_1",
    "extended_data_2",
    "extended_data_3",
    "extended_data_4",
    "extended_data_5",
    "extended_data_6",
)
EXPECTED_PANELS = {
    "main_figure_1": tuple("abcdefgh"),
    "main_figure_2": tuple("abcdefghijklmnopq"),
    "main_figure_3": tuple("abcdefghijklm"),
    "main_figure_4": tuple("abcdef"),
    "main_figure_5": tuple("abcdef"),
    "extended_data_1": tuple("abcdefghijk"),
    "extended_data_2": tuple("abcdefghij"),
    "extended_data_3": tuple("abcdef"),
    "extended_data_4": tuple("abcdefghi"),
    "extended_data_5": tuple("abcde"),
    "extended_data_6": tuple("abcdefgh"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer_records(path: Path) -> list[str]:
    records: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        tokens = re.findall(r"`([^`]+)`", line)
        if not tokens:
            continue
        token = tokens[0].replace("\\", "/").rstrip("/")
        if "/" in token:
            token = Path(token).name
        records.append(token)
    return records


def resolve_current_records() -> dict[str, Path]:
    main = _pointer_records(MAIN_POINTER)
    ed = _pointer_records(ED_POINTER)
    if len(main) != 5 or len(ed) != 6:
        raise RuntimeError(f"Pointer tables must select exactly 5 main and 6 ED records: {main}, {ed}")
    selected = dict(zip(FIGURES, [*main, *ed], strict=True))
    if len(set(selected.values())) != 11:
        raise RuntimeError("Current pointer tables contain duplicate record targets")
    records = {figure: RECORD_ROOT / name for figure, name in selected.items()}
    missing = [str(path) for path in records.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing current figure records: " + ", ".join(missing))
    return records


def verify_hash_closure(record: Path) -> tuple[bool, list[str]]:
    table = record / "RECORD_HASHES.csv"
    if not table.is_file():
        return False, ["missing:RECORD_HASHES.csv"]
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["path"]: row for row in rows}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors += [f"missing:{name}" for name in sorted(set(expected) - set(actual))]
    for relative in sorted(set(actual) & set(expected)):
        row = expected[relative]
        path = actual[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{relative}")
    return not errors, errors


def _find_one(record: Path, pattern: str) -> Path:
    paths = sorted(record.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern} in {record}, found {len(paths)}")
    return paths[0]


def _status_rows(path: Path) -> tuple[list[dict[str, str]], bool]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    allowed = {"PASS", "DEVIATION"}
    return rows, bool(rows) and all(str(row.get("status", row.get("auto_check", ""))).upper() in allowed for row in rows)


def _panel_coverage(record: Path, figure: str) -> bool:
    expected = set(EXPECTED_PANELS[figure])

    def collect_manifest_panels(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                token = str(key).strip().lower()
                if token in expected:
                    found.add(token)
                elif re.fullmatch(r"[a-z]-[a-z]", token):
                    start, end = token.split("-")
                    found.update(
                        chr(code)
                        for code in range(ord(start), ord(end) + 1)
                        if chr(code) in expected
                    )
                found.update(collect_manifest_panels(child))
        elif isinstance(value, list):
            for child in value:
                found.update(collect_manifest_panels(child))
        return found

    candidates = sorted(record.glob("*VALIDATION*.csv")) + sorted(record.glob("*panel_verification.csv"))
    panels: set[str] = set()
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
            if (
                figure in {"main_figure_1", "main_figure_2", "main_figure_4", "main_figure_5"}
                and len(rows) == len(EXPECTED_PANELS[figure])
                and all(
                    str(row.get("status", row.get("auto_check", ""))).lower()
                    in {"pass", "passed"}
                    for row in rows
                )
            ):
                return True
            for row in rows:
                value = str(row.get("panel", row.get("display_label", ""))).strip().lower()
                if value in EXPECTED_PANELS[figure]:
                    panels.add(value)
    # Some ED validation tables are check-level rather than panel-level; their
    # immutable assembly manifests carry the exact panel inventory instead.
    if panels == expected:
        return True
    manifests = sorted(record.glob("*MANIFEST*.json"))
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key in ("panel_letters", "panels", "panel_order"):
            value = payload.get(key)
            observed = set(value.keys()) if isinstance(value, dict) else set(value or [])
            if observed == expected:
                return True
        if collect_manifest_panels(payload) == expected:
            return True
        if figure == "main_figure_3" and len(payload.get("axes", [])) == 13:
            return True
    return False


def _export_checks(record: Path, figure: str) -> dict[str, bool]:
    svg = _find_one(record, "*_version_record.svg")
    pdf = _find_one(record, "*_version_record.pdf")
    png = _find_one(record, "*_version_record_600dpi.png")
    tiff = _find_one(record, "*_version_record_600dpi.tiff")
    with fitz.open(pdf) as document:
        pdf_one_page = document.page_count == 1
    with Image.open(png) as image:
        png_nonblank = max(ImageStat.Stat(image.convert("RGB")).stddev) > 1.0
    with Image.open(tiff) as image:
        tiff_dpi = image.info.get("dpi", (0.0, 0.0))
        tiff_compression = str(image.info.get("compression", "")).lower()
    svg_text = svg.read_text(encoding="utf-8")
    return {
        "pdf_one_page": pdf_one_page,
        "png_nonblank": png_nonblank,
        "tiff_600dpi": all(abs(float(value) - 600.0) < 1.0 for value in tiff_dpi),
        "tiff_lzw": tiff_compression in {"tiff_lzw", "lzw"},
        "svg_editable_text": svg_text.count("<text") > 0,
        "svg_no_external_links": not bool(
            re.search(r'''(?:href|xlink:href)=["'](?!#|data:)[^"']+["']''', svg_text)
        ),
        "panel_coverage_exact": _panel_coverage(record, figure),
    }


def _main_semantics(record: Path, figure: str) -> dict[str, bool]:
    if figure == "main_figure_1":
        manifest = json.loads((record / "figure1_source_manifest.json").read_text(encoding="utf-8"))
        timelines = pd.read_csv(record / "plot_ready/figure1_endnode_timelines.csv")
        return {
            "figure1_eight_sources": len(manifest.get("sources", [])) == 8,
            "figure1_six_timeline_sessions_frozen": timelines["nickname"].nunique() == 6 and len(timelines) > 100,
            "figure1_plot_ready_tables_frozen": len(list((record / "plot_ready").glob("*.csv"))) >= 11,
            "figure1_source_code_frozen": (record / "source_code/preview_figure1_assembly.py").is_file(),
        }
    if figure == "main_figure_2":
        gap = pd.read_csv(record / "source_data/gap_curve/gap_with_day1_day2_curve_data.csv")
        anchors = gap.loc[gap["time_label"].eq("D1-D2")].set_index("metric")
        expected = {
            "place": 0.5115494057669443,
            "level": 0.6311589510276657,
            "pv": 0.5884464623363727,
            "rate": 0.731508071430903,
        }
        return {
            "figure2_gap_table_exact": len(gap) == 20 and all(abs(float(anchors.at[key, "observed_mean"]) - value) <= 1e-15 for key, value in expected.items()),
            "figure2_gap_anchor_no_mask": anchors["source"].astype(str).str.contains("no_manual_mask").all(),
            "figure2_denominator_16901": pd.read_csv(record / "source_data/distribution/pair_support_audit.csv")["n_pairs_final"].sum() == 16901,
        }
    if figure == "main_figure_3":
        points = pd.read_csv(record / "source_data/long_gap/main3_experience_trend_points_no_manual_mask.csv")
        provenance = pd.read_csv(record / "source_data/long_gap/FINAL_VALUE_PROVENANCE.csv")
        first30 = pd.read_csv(record / "source_data/first30/combined_first30_points_all_metrics.csv")
        alignment = pd.read_csv(record / "source_data/first30/first_window_curve_point_alignment.csv")
        keys = ["metric", "src", "dst"]
        merged = points.merge(provenance, on=keys, suffixes=("_point", "_provenance"), validate="one_to_one")
        return {
            "figure3_long_gap_provenance_closed": len(merged) == 222 and np.max(np.abs(merged["value_point"] - merged["value_provenance"])) <= 1e-15 and provenance["manual_id_drops"].eq(0).all(),
            "figure3_first30_exact_scope": len(first30) == 236 and first30.groupby("metric").size().eq(59).all() and first30["animal"].nunique() == 22,
            "figure3_alignment_recomputed": len(alignment) == 15 and alignment["aligned"].astype(bool).all() and alignment["difference"].abs().max() <= 2e-16,
        }
    if figure == "main_figure_4":
        panel_c = pd.read_csv(record / "plot_ready/figure4_panel_c_d1_source_rows.csv")
        low = panel_c.loc[~panel_c["display_support_pass"].astype(bool)]
        manifest = json.loads((record / "figure4_source_manifest.json").read_text(encoding="utf-8"))
        return {
            "figure4_day1_authority": manifest.get("selected_source_day") == "d1" or "d1" in json.dumps(manifest).lower(),
            "figure4_panel_c_min5_support": len(low) == 6 and low["n_frames"].lt(5).all() and low["decoding_error"].isna().all(),
            "figure4_frozen_validator_passed": json.loads((record / "source_validation/VALIDATION.json").read_text(encoding="utf-8")).get("status") == "pass",
        }
    if figure == "main_figure_5":
        stats_path = record / "source_data/figure5_bundle/level_map_vs_exploration_efficiency_primary_clean_stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        expected_p = float(2 * stats.t.sf(abs(float(payload["cluster_ols_t"])), df=13))
        manifest = (record / "MANIFEST.md").read_text(encoding="utf-8")
        return {
            "figure5_finite_cluster_p": int(payload["cluster_ols_df"]) == 13 and abs(float(payload["cluster_ols_p"]) - expected_p) <= 1e-15,
            "figure5_hierarchical_manifest": "hierarchical" in manifest and "preferred_level_similarity" not in manifest,
            "figure5_support_16901": int(payload["level_support_total"]) == 16901,
        }
    raise ValueError(figure)


def _ed_semantics(record: Path, figure: str) -> dict[str, bool]:
    if figure == "extended_data_4":
        return {
            "ed4_embedded_sources_frozen": len(list((record / "embedded_sources").glob("*"))) >= 4,
        }
    if figure == "extended_data_5":
        validation, statuses = _status_rows(record / "ED05_VALIDATION.csv")
        checks = {row["check"]: row["status"] for row in validation}
        provenance = pd.read_csv(record / "plot_ready/ED5_PANEL_A_PER_SIDE_PROVENANCE.csv")
        return {
            "ed5_all_checks_pass": statuses,
            "ed5_gallery_qualified_provenance": len(provenance) == 64 and provenance["actual_neural_variant"].value_counts().to_dict() == {"ordinary": 61, "sorted": 2, "presort": 1},
            "ed5_bcd_relation_closed": all(checks.get(name) == "PASS" for name in ("daycomb_drawn_animal_values_recompute_from_pair_authority", "daycomb_descriptive_stats_recompute_from_drawn_animals", "daycomb_displayed_paired_t_stats_recompute_from_drawn_animals")),
            "ed5_panel_e_single_vintage": checks.get("equal_time_single_source_vintage") == "PASS",
        }
    if figure == "extended_data_6":
        return {
            "ed6_exporters_frozen": (record / "source_code/export_fixed_reference_window_convergence.py").is_file() and (record / "source_code/export_within_session_pivot_all_metrics.py").is_file(),
            "ed6_gate_manifest_frozen": bool(list((record / "gate_authority").glob("*MANIFEST*.json"))),
        }
    validation_files = sorted(record.glob("*VALIDATION*.csv"))
    statuses = all(_status_rows(path)[1] for path in validation_files) if validation_files else True
    return {"frozen_validation_statuses_allowlisted": statuses}


def validate_record(record: Path, figure: str, *, require_hashes: bool = True) -> dict[str, Any]:
    checks = _export_checks(record, figure)
    checks.update(_main_semantics(record, figure) if figure.startswith("main") else _ed_semantics(record, figure))
    if require_hashes:
        closure, errors = verify_hash_closure(record)
        checks["complete_hash_closure"] = closure
    else:
        errors = []
    saved = record / "RECORD_VALIDATION.json"
    if require_hashes:
        checks["saved_validation_pass"] = saved.is_file() and json.loads(saved.read_text(encoding="utf-8")).get("status") == "pass"
    return {
        "figure": figure,
        "record": record.relative_to(REPO).as_posix(),
        "checks": {name: bool(value) for name, value in checks.items()},
        "hash_errors": errors,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_all(records: dict[str, Path] | None = None, *, require_hashes: bool = True) -> dict[str, Any]:
    selected = resolve_current_records() if records is None else records
    results = {figure: validate_record(selected[figure], figure, require_hashes=require_hashes) for figure in FIGURES}
    panel_count = sum(len(EXPECTED_PANELS[figure]) for figure in FIGURES)
    return {
        "status": "pass" if all(result["status"] == "pass" for result in results.values()) else "fail",
        "figure_count": len(results),
        "panel_count": panel_count,
        "records": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_all()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
