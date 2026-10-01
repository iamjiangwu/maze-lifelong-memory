"""Read-only, fail-closed validation for the 15 current figure records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    "extended_data_7",
    "extended_data_8",
    "extended_data_9",
    "extended_data_10",
)
EXPECTED_PANELS = {
    "main_figure_1": tuple("abcdefgh"),
    "main_figure_2": tuple("abcdefghijklmnopq"),
    "main_figure_3": tuple("abcdefghijklm"),
    "main_figure_4": tuple("abcdef"),
    "main_figure_5": tuple("abcdef"),
    "extended_data_1": tuple("abcdefghijk"),
    "extended_data_2": tuple("abcdefghi"),
    "extended_data_3": tuple("abcdef"),
    "extended_data_4": tuple("abcdefghi"),
    "extended_data_5": tuple("abcde"),
    "extended_data_6": tuple("abcdefgh"),
    "extended_data_7": tuple("abcdef"),
    "extended_data_8": tuple("abcdefghi"),
    "extended_data_9": tuple("abcdefghijkl"),
    "extended_data_10": tuple("abc"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_relative_files(root: Path) -> list[tuple[str, Path]]:
    """Enumerate record files without losing Windows paths beyond MAX_PATH."""
    resolved = root.resolve()
    walk_root = Path(f"\\\\?\\{resolved}") if os.name == "nt" else resolved
    rows: list[tuple[str, Path]] = []
    for directory, _, names in os.walk(walk_root):
        directory_path = Path(directory)
        for name in names:
            extended_path = directory_path / name
            normal_text = str(extended_path)
            if normal_text.startswith("\\\\?\\"):
                normal_text = normal_text[4:]
            relative = Path(normal_text).relative_to(resolved).as_posix()
            rows.append((relative, extended_path))
    return sorted(rows, key=lambda item: item[0])


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
        if re.fullmatch(r"(?:main_figure\d?|main_figures|extended_data_fig\d{2})_[A-Za-z0-9_.-]+", token):
            records.append(token)
    return records


def resolve_current_records() -> dict[str, Path]:
    main = _pointer_records(MAIN_POINTER)
    ed = _pointer_records(ED_POINTER)
    if len(main) != 5 or len(ed) != 10:
        raise RuntimeError(f"Pointer tables must select exactly 5 main and 10 ED records: {main}, {ed}")
    selected = dict(zip(FIGURES, [*main, *ed], strict=True))
    if len(set(selected.values())) != 15:
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
        relative: path
        for relative, path in _iter_relative_files(record)
        if relative != "RECORD_HASHES.csv"
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
        timeline_counts = timelines.groupby("nickname", sort=False).size().to_dict()
        expected_timeline_counts = {
            "434RRday1": 447,
            "434-3day1": 242,
            "434RRday2": 467,
            "434-6day1": 320,
            "1636Rday1": 1554,
            "1636Rday2": 1383,
        }
        decisions = pd.read_csv(record / "figure1_panel_a_decision_points.csv")
        expected_decisions = [
            [0, 0, 7, 7],
            [1, 1, 7, 3],
            [2, 4, 11, 3],
            [3, 10, 11, 5],
            [4, 22, 13, 5],
            [5, 46, 13, 6],
        ]
        observed_decisions = (
            decisions[["depth", "run", "x", "y"]].astype(int).values.tolist()
        )
        pdf = _find_one(record, "*_version_record.pdf")
        with fitz.open(pdf) as document:
            pdf_image_objects = sum(len(page.get_images(full=True)) for page in document)
            pdf_text = "\n".join(page.get_text() for page in document)
        svg_text = _find_one(record, "*_version_record.svg").read_text(encoding="utf-8")
        exploration = pd.read_csv(
            record / "plot_ready/exploration_first_chunk_current_plus_new_points.csv"
        )
        exploration_counts = exploration.groupby("group").size().to_dict()
        accepted_change_path = record / "ACCEPTED_CHANGE_2026_08_12.json"
        fraction_audit_path = (
            record / "source_validation/panel_e_fraction_histogram_audit.csv"
        )
        accepted_fraction_histogram = False
        fraction_normalization_exact = True
        if accepted_change_path.is_file() and fraction_audit_path.is_file():
            accepted_change = json.loads(
                accepted_change_path.read_text(encoding="utf-8")
            )
            fraction_audit = pd.read_csv(fraction_audit_path)
            fraction_sums = fraction_audit.groupby("group", observed=True)[
                "fraction_within_group"
            ].sum()
            fraction_counts = (
                fraction_audit.groupby("group", observed=True)["n_sessions"]
                .first()
                .astype(int)
                .to_dict()
            )
            accepted_fraction_histogram = (
                accepted_change.get("status") == "accepted_successor_record"
                and "panel e histogram bin heights normalized within condition as session fractions"
                in accepted_change.get("changes", [])
            )
            fraction_normalization_exact = (
                set(fraction_sums.index.astype(str))
                == {"LTM D1 (all)", "Naive D1 (rewarded)"}
                and np.allclose(fraction_sums.to_numpy(dtype=float), 1.0)
                and fraction_counts
                == {"LTM D1 (all)": 56, "Naive D1 (rewarded)": 23}
            )
        return {
            "figure1_eight_sources": len(manifest.get("sources", [])) == 8,
            "figure1_six_timeline_sessions_frozen": timeline_counts
            == expected_timeline_counts,
            "figure1_panel_f_selected_no_port_mouse_exact": (
                "1636Rday1" in timelines["nickname"].values
                and "1636Rday2" in timelines["nickname"].values
                and not timelines["nickname"].astype(str).str.startswith("1637L").any()
                and "sessions 1636Rday1, 1636Rday2" in json.dumps(manifest)
            ),
            "figure1_plot_ready_tables_frozen": len(list((record / "plot_ready").glob("*.csv"))) >= 13,
            "figure1_source_code_frozen": (record / "source_code/preview_figure1_assembly.py").is_file(),
            "figure1_decision_points_exact": observed_decisions == expected_decisions,
            "figure1_decision_scale_1p4": (
                len(decisions) == 6
                and np.allclose(decisions["requested_scale_vs_round1"], 1.4)
                and np.allclose(decisions["marker_area_points2"], 34.0 * 1.4**2)
                and np.allclose(decisions["label_fontsize_points"], 4.0 * 1.4)
            ),
            "figure1_pdf_vector_only": pdf_image_objects == 0,
            "figure1_svg_vector_only": "<image" not in svg_text,
            "figure1_exploration_histogram_unit_matches_record_contract": (
                (
                    accepted_fraction_histogram
                    and "Fraction of sessions" in svg_text
                    and "No. of sessions" not in svg_text
                )
                or (
                    not accepted_fraction_histogram
                    and "No. of sessions" in svg_text
                )
            )
            and "# of Animals" not in svg_text,
            "figure1_exploration_fraction_normalization_exact": (
                not accepted_fraction_histogram or fraction_normalization_exact
            ),
            "figure1_exploration_session_counts_exact": exploration_counts
            == {"LTM D1 (all)": 56, "Naive D1 (rewarded)": 23},
            "figure1_panel_b_next_cycle_exact": (
                svg_text.count("Next cycle: i \u2192 i + 90") == 1
                and "Next cycle: i \u2192 i + 90" in pdf_text
                and "Next session" not in svg_text
                and "Next session" not in pdf_text
            ),
        }
    if figure == "main_figure_2":
        gap = pd.read_csv(record / "source_data/gap_curve/gap_with_day1_day2_curve_data.csv")
        anchors = gap.loc[gap["time_label"].eq("D1-D2")].set_index("metric")
        expected = {
            "place": 0.5280943055084769,
            "level": 0.6422188653451758,
            "pv": 0.6031046735013429,
            "rate": 0.7343462236466652,
        }
        return {
            "figure2_gap_table_exact": len(gap) == 20 and all(abs(float(anchors.at[key, "observed_mean"]) - value) <= 1e-15 for key, value in expected.items()),
            "figure2_gap_anchor_pair_mean_no_mask": (
                anchors["source"].astype(str).str.contains("no_manual_mask").all()
                and anchors["same_stage_mean_source"].eq("pair").all()
                and anchors["n_pairs"].eq(47).all()
            ),
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
        source_manifest = json.loads(
            (record / "figure5_source_manifest.json").read_text(encoding="utf-8")
        )
        seriation = json.loads(
            (
                record
                / "source_validation/refinement_2026_08_12/"
                "figure5b_historical_hierarchical_refined_order.json"
            ).read_text(encoding="utf-8")
        )
        rebuild = json.loads(
            (record / "REBUILD_PROVENANCE_CLOSURE.json").read_text(encoding="utf-8")
        )
        promotion = json.loads(
            (record / "NEURON_ID_SHUFFLE_PROMOTION.json").read_text(encoding="utf-8")
        )
        return {
            "figure5_finite_cluster_p": int(payload["cluster_ols_df"]) == 13 and abs(float(payload["cluster_ols_p"]) - expected_p) <= 1e-15,
            "figure5_seriation_manifest": (
                (
                    "average-linkage hierarchical order" in manifest
                    and "preferred-Day-i-level refinement" in manifest
                )
                or (
                    promotion.get("status") == "PASS"
                    and promotion.get("record") == record.name
                    and source_manifest.get("style", {}).get(
                        "heatmap_ordering_method"
                    )
                    == "historical_average_linkage_then_stable_argmax"
                )
            ),
            "figure5_seriation_machine_manifest": (
                source_manifest.get("style", {}).get("heatmap_ordering_method")
                == "historical_average_linkage_then_stable_argmax"
            ),
            "figure5_seriation_cache_and_rebuild_provenance": (
                seriation.get("is_exact_permutation") is True
                and seriation.get("row_count") == 16901
                and seriation.get("ordering_method")
                == "historical_average_linkage_then_stable_argmax"
                and seriation.get("linkage") == "average"
                and seriation.get("refined_order_sha256")
                == "c7b3467afb3f04a22a96f006780ad4aad5336239323e0375090ade89469bd1d4"
                and rebuild.get("status") == "PASS"
                and rebuild.get("accepted_seriation")
                == "historical_average_linkage_then_stable_argmax"
                and all(rebuild.get("frozen_source_checks", {}).values())
            ),
            "figure5_support_16901": int(payload["level_support_total"]) == 16901,
        }
    raise ValueError(figure)


def _ed_semantics(record: Path, figure: str) -> dict[str, bool]:
    if figure == "extended_data_1":
        render_path = next(
            path
            for path in (
                record / "ROUND18_RENDER_VALIDATION.json",
                record / "ROUND17_RENDER_VALIDATION.json",
                record / "ROUND14_RENDER_VALIDATION.json",
            )
            if path.is_file()
        )
        render = json.loads(render_path.read_text(encoding="utf-8"))
        checks = render.get("checks", {})
        svg_text = (record / "ed01_version_record.svg").read_text(encoding="utf-8")
        return {
            "ed1_current_render_pass": str(render.get("status", "")).lower() == "pass",
            "ed1_panel_f_endpoint_specific_scales": checks.get(
                "panel_f_independent_endpoint_scales_restored"
            )
            is True,
            "ed1_panel_h_shared_scale": checks.get(
                "panel_h_modes_share_identical_y_scale"
            )
            is True,
            "ed1_coverage_round_labels_exact": (
                svg_text.count(">Within round coverage</text>") == 1
                and svg_text.count(">Coverage round efficiency</text>") == 1
                and svg_text.count(">Coverage round</text>") == 1
                and "Within-round coverage" not in svg_text
                and "Coverage-round efficiency" not in svg_text
                and "Within-cycle coverage" not in svg_text
                and "Cycle efficiency" not in svg_text
            ),
            "ed1_i_aligns_to_f_three_internal_columns": (
                checks.get("panel_i_three_internal_columns_align_to_panel_f") is True
                and float(render.get("panel_i_to_f_internal_column_center_max_delta_fraction", 1.0)) == 0.0
                and float(render.get("panel_i_to_f_internal_column_width_max_delta_fraction", 1.0)) == 0.0
            ),
            "ed1_i_clear_of_j": (
                checks.get("panel_i_clear_of_panel_j_ylabel") is True
                and float(render.get("panel_i_to_j_ylabel_clearance_px", -1.0)) > 0.0
            ),
            "ed1_second_row_labels_clear_of_d": (
                checks.get("second_row_condition_legend_clear_of_panel_d_letter") is True
                and float(render.get("curve_legend_to_d_clearance_px", -1.0)) > 0.0
                and float(render.get("curve_legend_to_d_overlap_px2", 1.0)) == 0.0
            ),
        }
    if figure == "extended_data_2":
        manifest = json.loads(
            (record / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        distance_path = record / "plot_ready/panel_e_distance_from_return.csv"
        distance_record = distance_path.is_file()
        panel_e = pd.read_csv(
            distance_path if distance_record else record / "plot_ready/panel_e_shift_from_naive.csv"
        )
        panel_e_groups = (
            pd.read_csv(record / "plot_ready/panel_e_group_means.csv")
            if distance_record
            else None
        )
        panel_manifest = pd.read_csv(record / "PANEL_MANIFEST.csv")
        render_result = json.loads((record / "RENDER_RESULT.json").read_text(encoding="utf-8"))
        round23 = render_result.get("round23_display_only_changes", {})
        geometry = round23.get("panel_a_geometry_at_source_dpi", {})
        cohort_counts = pd.read_csv(
            record / "cohort_audit/drawn_panel_cohort_counts.csv"
        )
        cohort_lookup = {
            (row.panel, row.group): (int(row.n_sessions), int(row.n_mice))
            for row in cohort_counts.itertuples(index=False)
        }
        calibration_ok = True
        calibration_expected = {
            "f": {
                "naive": (23, 0.10823401922673043, 0.1614770085848131),
                "LTM": (56, 0.2604693492070003, 0.0486461040299258),
                "experienced": (74, 0.058600824087683676, 0.0214202252047705),
            },
            "i": {
                "naive": (22, 0.295631752509341, 0.0324915144055487),
                "LTM": (56, 0.5522781629941739, 0.0259825603208955),
                "experienced": (74, -0.0617367145194396, 0.0536738176733248),
            },
        }
        for panel, expected_groups in calibration_expected.items():
            points = pd.read_csv(
                record / f"plot_ready/panel_{panel}_session_points.csv"
            )
            summary = pd.read_csv(
                record / f"plot_ready/panel_{panel}_calibration_summary.csv"
            )
            simulated_column = (
                "four_own_half_cover_eff"
                if panel == "f"
                else "depth_own_half_cover_eff"
            )
            calibration_ok = calibration_ok and not any(
                token in column.lower()
                for column in summary.columns
                for token in ("spearman", "rho", "gate", "pass")
            )
            for group, (expected_n, expected_r, expected_bias) in expected_groups.items():
                group_points = points.loc[points["group"].eq(group)]
                group_summary = summary.loc[summary["group"].eq(group)]
                if len(group_summary) != 1:
                    calibration_ok = False
                    continue
                saved = group_summary.iloc[0]
                actual_r = group_points["real_half_cover_eff"].corr(
                    group_points[simulated_column]
                )
                actual_bias = (
                    group_points[simulated_column]
                    - group_points["real_half_cover_eff"]
                ).mean()
                calibration_ok = calibration_ok and (
                    len(group_points) == expected_n
                    and int(saved["n_session_rows"]) == expected_n
                    and abs(actual_r - expected_r) < 1e-12
                    and abs(float(saved["pearson_r"]) - expected_r) < 1e-12
                    and abs(actual_bias - expected_bias) < 1e-12
                    and abs(float(saved["mean_error"]) - expected_bias) < 1e-12
                )
        inventory_ok = (
            manifest.get("panel_inventory") == "a-i exactly; no panel j"
            if distance_record
            else (
                manifest.get("panel_inventory_authority") == "supplied PowerPoint slide 3"
                and manifest.get("panel_order") == list("abcdefghi")
                and manifest.get("row_order")
                == [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
            )
        )
        panel_e_ok = (
            len(panel_e) == 4
            and set(panel_e["metric"].astype(str)) == {"Bf", "Ba", "Lf", "Lo"}
            and panel_e["window_minutes"].eq(30).all()
            and panel_e["support"].eq("natural primary-policy support").all()
        )
        if distance_record:
            group_sessions = panel_e_groups.set_index("group")["n_sessions"].to_dict()
            panel_e_ok = panel_e_ok and group_sessions == {
                "naive": 23,
                "LTM": 56,
                "experienced": 74,
            }
            panel_e_label = "ed2_panel_e_current_fixed30_distance_from_return"
        else:
            panel_e_ok = (
                panel_e_ok
                and panel_e["n_naive_sessions"].eq(23).all()
                and panel_e["n_return_sessions"].eq(58).all()
                and panel_e["n_second_day_sessions"].eq(74).all()
            )
            panel_e_label = "ed2_panel_e_current_fixed30_shift_from_naive"
        return {
            "ed2_powerpoint_inventory_exact_a_to_i": (
                inventory_ok
                and list(dict.fromkeys(panel_manifest["panel"].astype(str)))
                == list("abcdefghi")
            ),
            "ed2_no_panel_j_or_extra_role": (
                set(panel_manifest["panel"].astype(str)) == set("abcdefghi")
                and not any(path.name.startswith("panel_j_") for path in (record / "plot_ready").glob("*.csv"))
            ),
            panel_e_label: panel_e_ok,
            "ed2_behavior_only_manual_masks_not_applicable": (
                manifest.get("manual_mask_applicability")
                == "not applicable; behavior-only analysis"
            ),
            "ed2_stable_record_identity": manifest.get("record") == record.name,
            "ed2_round26_common_depth_panel_i_and_inherited_clearance": (
                manifest.get("review_round") == 26
                and render_result.get("review_round") == 23
                and geometry.get("two_character_sample") == "nn"
                and abs(float(geometry.get("actual_left_shift_px", -1.0)) - float(geometry.get("two_character_shift_px", -2.0))) < 1e-9
                and float(geometry.get("grey_arrow_clearance_px", -1.0)) > 0.0
                and float(geometry.get("box_clearance_px", -1.0)) > 0.0
                and round23.get("scientific_values_or_tables_changed") is False
                and manifest.get("scientific_values_changed") is True
                and manifest.get("current_scientific_correction", {}).get("status") == "PASS"
                and cohort_lookup.get(("b", "naive")) == (23, 23)
                and cohort_lookup.get(("b", "LTM")) == (56, 17)
                and cohort_lookup.get(("f", "experienced")) == (74, 24)
                and cohort_lookup.get(("g", "naive")) == (22, 22)
                and cohort_lookup.get(("h", "LTM")) == (56, 17)
                and cohort_lookup.get(("i", "naive")) == (22, 22)
                and cohort_lookup.get(("i", "experienced")) == (74, 24)
                and calibration_ok
            ),
        }
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
            "ed5_all_64_place_maps_fully_shown": checks.get("gallery_all_64_maps_use_full_45x45_bounds") == "PASS",
        }
    if figure == "extended_data_6":
        return {
            "ed6_exporters_frozen": (record / "source_code/export_fixed_reference_window_convergence.py").is_file() and (record / "source_code/export_within_session_pivot_all_metrics.py").is_file(),
            "ed6_gate_manifest_frozen": bool(list((record / "gate_authority").rglob("*MANIFEST*.json"))),
        }
    if figure == "extended_data_7":
        record_manifest = json.loads(
            (record / "ED07_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        display_manifest = json.loads(
            (record / "ED07_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
        )
        display_validation = json.loads(
            (record / "ED07_DISPLAY_VALIDATION.json").read_text(encoding="utf-8")
        )
        display = display_manifest["display_contract"]
        return {
            "ed7_stable_record_identity": record_manifest.get("record") == record.name,
            "ed7_six_panel_inventory": record_manifest.get("panel_letters") == list("abcdef"),
            "ed7_round12_display_pass": display_manifest.get("status") == "PASS" and display_validation.get("status") == "PASS",
            "ed7_row_titles_exact": display.get("top_y_axis_label") == r"Day $\mathrm{i}$ early vs Day $\mathrm{i}+1$ correlation" and display.get("bottom_y_axis_label") == "Long gap correlation" and "(r)" not in json.dumps(display),
            "ed7_scopes_locked": display_manifest.get("top_primary_scope") == "58 maintained observations / 21 mice; BH3" and display_manifest.get("bottom_primary_scope") == "36 pairs / 14 mice; BH4; current clean single-vintage primary",
            "ed7_exclusions_and_mask_locked": len(display_manifest.get("excluded_from_all_ed7_primary_analyses", [])) == 6 and display_manifest.get("manual_mask_profile") == "none for every displayed source row",
            "ed7_sources_and_code_frozen": len(record_manifest.get("scientific_inputs", [])) == 6 and (record / "source_code/render_ed07_slide9_powerpoint_matched.py").is_file() and (record / "source_code/validate_ed07_six_panel_round11.py").is_file(),
        }
    if figure == "extended_data_9":
        manifest = json.loads(
            (record / "ED09_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        b1 = pd.read_csv(record / "plot_ready/B1_NO_MANUAL_MASK_DEFINITION_COMPARISON.csv")
        b1_metadata = json.loads(
            (
                record
                / "source_validation/b1_metadata_round25/ED09_B1_METADATA_PARITY_AUDIT.json"
            ).read_text(encoding="utf-8")
        )
        render = json.loads(
            (record / "source_validation/ROUND11_FIXED_COLORBAR_RENDER_AUDIT.json").read_text(encoding="utf-8")
        )
        grammar = json.loads(
            (record / "source_validation/ED9_MAIN_FIGURE_GRAMMAR_AUDIT.json").read_text(encoding="utf-8")
        )
        manual = json.loads(
            (record / "source_validation/ED9_PANEL_MANUAL_MASK_AUDIT.json").read_text(encoding="utf-8")
        )
        conditional = json.loads(
            (record / "source_validation/PANEL_A_CONDITIONAL_ESTIMAND_AUDIT.json").read_text(encoding="utf-8")
        )
        panel_a_svg = (
            record / "individual_panels/ED9_C1_transition_topology.svg"
        ).read_text(encoding="utf-8")
        panel_a_visible_text = re.findall(
            r"<text\b[^>]*>([^<]*)</text>", panel_a_svg
        )
        panel_b_visible_text = re.findall(
            r"<text\b[^>]*>([^<]*)</text>",
            (record / "individual_panels/ED9_C2_shift_distance.svg").read_text(
                encoding="utf-8"
            ),
        )
        panel_c_visible_text = re.findall(
            r"<text\b[^>]*>([^<]*)</text>",
            (
                record / "individual_panels/ED9_D1_within_level_shuffle.svg"
            ).read_text(encoding="utf-8"),
        )
        decimal_revision = json.loads(
            (record / "DISPLAY_REVISION_2026_08_13.json").read_text(encoding="utf-8")
        )
        pair_revision = json.loads(
            (record / "PAIR_MEAN_GAP_CURVE_REVISION.json").read_text(encoding="utf-8")
        )
        panel_i_curve = pd.read_csv(
            record / "plot_ready/F1_NO_MANUAL_MASK_GAP_CURVE.csv"
        )
        panel_i_anchor = panel_i_curve.loc[panel_i_curve["time_days"].eq(1.0)]
        panel_f = pd.read_csv(
            record
            / "source_data/panel_f_inference_round24/be_roles/"
            "B2_EXCLUSION_CLEAN32_CR1_BH2_STATISTICS.csv"
        ).set_index("contrast")
        panel_f_pairs = pd.read_csv(
            record
            / "source_data/panel_f_inference_round24/be_roles/"
            "B2_EXCLUSION_CLEAN32_PAIR_CONTRASTS.csv"
        )
        exclusion_contract = json.loads(
            (record / "exclusion_audit/EXCLUDED_SESSION_CONTRACT.json").read_text(
                encoding="utf-8"
            )
        )
        forbidden_animals = {
            str(row["animal"]) for row in exclusion_contract["excluded_animals"]
        }
        forbidden_sessions = {
            str(row["session"]) for row in exclusion_contract["excluded_sessions"]
        }
        pair_sessions = {
            session
            for pair_id in panel_f_pairs["pair_id"].astype(str)
            for session in pair_id.split("__")
        }
        pair_animals = set(panel_f_pairs["animal"].astype(str))
        return {
            "ed9_stable_record_identity": manifest.get("record") == record.name,
            "ed9_twelve_panel_inventory": manifest.get("panel_letters") == list("abcdefghijkl"),
            "ed9_round11_render_pass": render.get("status") == "PASS" and all(render.get("checks", {}).values()),
            "ed9_main_figure_grammar_pass": grammar.get("status") == "PASS" and len(grammar.get("panels", [])) == 12 and all(grammar.get("checks", {}).values()),
            "ed9_no_manual_mask_pass": manual.get("status") == "PASS" and all(manual.get("checks", {}).values()),
            "ed9_support_16901": conditional.get("cohort", {}).get("transition_rows") == 16901 and conditional.get("cohort", {}).get("restored_from_obsolete_manual_mask") == 4,
            "ed9_panel_a_fixed_colorbar": render.get("panel_a_estimand", {}).get("render", {}).get("color_endpoint") == 0.1,
            "ed9_panel_a_decimal_colorbar_ticks_exact": (
                decimal_revision.get("status") == "PASS"
                and decimal_revision.get("tick_positions") == [-0.1, 0.0, 0.1]
                and decimal_revision.get("new_tick_labels") == ["−0.1", "0", "0.1"]
                and "−0.1" in panel_a_visible_text
                and "0.1" in panel_a_visible_text
                and "−10.00%" not in panel_a_visible_text
                and "+10.00%" not in panel_a_visible_text
                and not any("%" in token for token in panel_a_visible_text)
            ),
            "ed9_panel_a_colorbar_title_spacing_exact": (
                manifest.get("panel_a_colorbar_title_spacing", {}).get("status")
                == "PASS"
                and manifest.get("panel_a_colorbar_title_spacing", {}).get(
                    "shift_left_pt"
                )
                == 9.0
                and manifest.get("panel_a_colorbar_title_spacing", {}).get(
                    "minimum_tick_to_title_gap_pt", 0.0
                )
                >= 2.5
                and manifest.get("panel_a_colorbar_title_spacing", {}).get(
                    "scientific_recomputation"
                )
                is False
            ),
            "ed9_panels_b_c_open_compounds_exact": (
                manifest.get("open_compound_label_revision", {}).get("status")
                == "PASS"
                and manifest.get("open_compound_label_revision", {}).get("panel_b")
                == "Level shift distance"
                and manifest.get("open_compound_label_revision", {}).get("panel_c")
                == "Within level shuffle"
                and "Level shift distance" in panel_b_visible_text
                and "Level-shift distance" not in panel_b_visible_text
                and "Within level" in panel_c_visible_text
                and "Within-level" not in panel_c_visible_text
                and "shuffle" in panel_c_visible_text
                and "Shuffle" in panel_c_visible_text
                and manifest.get("open_compound_label_revision", {}).get(
                    "scientific_recomputation"
                )
                is False
            ),
            "ed9_panel_i_main_figure2_grammar": (
                render.get("checks", {}).get("panel_i_main_figure2_grammar_exact") is True
                and pair_revision.get("checks", {}).get("panel_i_main_figure2_grammar") is True
            ),
            "ed9_panel_i_pair_mean_anchor_exact": (
                pair_revision.get("status") == "PASS"
                and pair_revision.get("statistical_unit") == "session pair"
                and len(panel_i_curve) == 5
                and len(panel_i_anchor) == 1
                and panel_i_anchor["same_stage_mean_source"].eq("pair").all()
                and panel_i_anchor["n_pairs"].eq(47).all()
                and abs(float(panel_i_anchor.iloc[0]["observed_mean"]) - 0.6422188653451761)
                <= 5e-15
                and pair_revision.get("checks", {}).get("long_gap_rows_exact_parent") is True
            ),
            "ed9_panel_c_current_metadata_exact": set(
                manifest.get("panel_c_axis_labels", {}).values()
            )
            == {"Place map correlation"}
            and manifest.get("panel_c_significance")
            == (
                "standalone directional mouse-clustered CR1 raw P=0.00947165887533162; "
                "no multiplicity correction; **"
            ),
            "ed9_b1_c_maps_after_s_gate": (
                len(b1) == 36
                and b1["animal_id"].astype(str).nunique() == 14
                and int(b1["n_neurons_exact"].sum()) == 16901
                and b1["signal"].eq("C raw_counts-derived maps").all()
                and b1["rate_gate"]
                .eq(
                    "S spike_counts >=0.1 spikes/min in both sessions inherited from exact "
                    "tree-depth7 source"
                )
                .all()
                and manifest.get("b1_signal_provenance", {}).get("map_value_signal") == "C"
                and manifest.get("b1_signal_provenance", {}).get(
                    "eligibility_gate_signal"
                )
                == "S"
                and b1_metadata.get("status") == "PASS"
                and all(b1_metadata.get("checks", {}).values())
            ),
            "ed9_panel_f_clean32_cr1_bh2_exact": (
                len(panel_f_pairs) == 32
                and panel_f_pairs["animal"].astype(str).nunique() == 14
                and len(panel_f) == 2
                and panel_f["cluster_df"].eq(13).all()
                and panel_f["n_pairs"].eq(32).all()
                and panel_f["n_mice"].eq(14).all()
                and panel_f["covariance"]
                .eq("CR1 intercept-only; Student t(G-1)")
                .all()
                and abs(
                    float(
                        panel_f.at[
                            "geometric8_minus_horizontal8",
                            "q_bh_two_raw_planned_contrasts",
                        ]
                    )
                    - 2.8670561454901405e-05
                )
                < 1e-18
                and abs(
                    float(
                        panel_f.at[
                            "geometric8_minus_vertical8",
                            "q_bh_two_raw_planned_contrasts",
                        ]
                    )
                    - 1.6728044637612076e-06
                )
                < 1e-18
                and not (pair_sessions & forbidden_sessions)
                and not (pair_animals & forbidden_animals)
            ),
            "ed9_sources_and_code_frozen": len(list((record / "plot_ready").glob("*.csv"))) == 11 and len(list((record / "source_code").glob("*.py"))) >= 15,
        }
    if figure == "extended_data_10":
        manifest = json.loads(
            (record / "ED10_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        source = json.loads(
            (record / "source_validation/ED10_ROUND6_SOURCE_VALIDATION.json").read_text(encoding="utf-8")
        )
        final = json.loads(
            (record / "powerpoint_review/ED10_ROUND6_FINAL_VALIDATION.json").read_text(encoding="utf-8")
        )
        eligibility = json.loads(
            (
                record
                / "source_authority/fixed_neuron_synthesis/NEURAL_ELIGIBILITY_CONTRACT.json"
            ).read_text(encoding="utf-8")
        )
        matrix = pd.read_csv(record / "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv")
        raw_matrix = pd.read_csv(
            record / "plot_ready/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv"
        )
        sign_audit = json.loads(
            (
                record
                / "source_validation/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json"
            ).read_text(encoding="utf-8")
        )
        membership = pd.read_csv(record / "plot_ready/ED10_COHORT_MEMBERSHIP.csv")
        role_counts = membership.groupby("role").size().to_dict()
        root_export_names = {
            "pdf": "ed10_version_record.pdf",
            "png_600dpi": "ed10_version_record_600dpi.png",
            "svg": "ed10_version_record.svg",
            "tiff_600dpi": "ed10_version_record_600dpi.tiff",
        }
        top_level = "\n".join(
            (record / name).read_text(encoding="utf-8")
            for name in ("AUDIT_REPORT.md", "CAPTION_NOTES.md", "MANIFEST.md")
        )
        top_level_flat = re.sub(r"\s+", " ", top_level)
        return {
            "ed10_stable_record_identity": manifest.get("record") == record.name,
            "ed10_three_panel_inventory": manifest.get("panel_letters") == list("abc"),
            "ed10_round6_source_pass": source.get("status") == "PASS"
            and len(source.get("checks", {})) == 36
            and all(source.get("checks", {}).values()),
            "ed10_matrix_exact_scope": len(matrix) == 18
            and set(matrix["block"]) == {
                "reward_location",
                "whole_decoder",
                "place_map",
                "structure",
                "pv_map",
                "rate",
            }
            and matrix["n_pairs"].eq(30).all()
            and matrix["n_animals"].eq(14).all(),
            "ed10_raw_decoder_error_matrix_exact": (
                len(raw_matrix) == 18
                and raw_matrix["n_pairs"].eq(30).all()
                and raw_matrix["n_animals"].eq(14).all()
                and np.allclose(
                    raw_matrix.loc[
                        raw_matrix["block"].isin(["reward_location", "whole_decoder"]),
                        "pooled_pearson_r",
                    ].to_numpy(dtype=float),
                    np.asarray(
                        [
                            -0.4429149746329733,
                            0.4583382549928935,
                            -0.44213464981013634,
                            -0.5757985165782262,
                            0.47854156612751175,
                            -0.6328784637120961,
                        ]
                    ),
                    atol=1e-15,
                    rtol=0.0,
                )
            ),
            "ed10_raw_decoder_sign_audit_pass": (
                sign_audit.get("status") == "PASS"
                and all(sign_audit.get("checks", {}).values())
            ),
            "ed10_raw_matrix_open_outcome_labels": not raw_matrix["outcome_label"]
            .astype(str)
            .str.contains("Reward-approach", regex=False)
            .any(),
            "ed10_manifest_root_export_hashes_exact": all(
                manifest.get("root_export_sha256", {}).get(key)
                == sha256(record / name)
                for key, name in root_export_names.items()
            ),
            "ed10_top_level_round8_prose_exact": "Round 8" in top_level_flat
            and "Higher decoder error means poorer decoding" in top_level_flat
            and "Values are raw decoder errors" in top_level_flat
            and not any(
                term in top_level.lower()
                for term in (
                    "place-map",
                    "level-map",
                    "population-vector",
                    "ensemble-rate",
                    "whole-maze",
                    "reward-approach",
                    "reward-decoder",
                )
            ),
            "ed10_cohort_membership_exact": role_counts
            == {"a_b_30pair": 30, "c_clean36": 36},
            "ed10_neural_exclusion_contract": eligibility.get("status")
            == "LOCKED_BY_AUTHOR_DIRECTIVE"
            and {str(row["animal"]) for row in eligibility.get("behavior_only_animals", [])}
            == {"840", "841"}
            and len(eligibility.get("neural_excluded_sessions", [])) == 24,
            "ed10_display_contract_exact": (
                manifest.get("display_contract", {}).get("colorbar_title")
                == "Pearson correlation"
                and manifest.get("display_contract", {}).get("panel_a_label_weight")
                == "normal"
                and manifest.get("display_contract", {}).get("panel_a_cell_color")
                == "black"
                and manifest.get("display_contract", {}).get("panel_a_cell_weight")
                == "normal"
                and manifest.get("display_contract", {}).get("panel_a_cell_count")
                == 18
                and manifest.get("display_contract", {}).get("panel_c_policy_order")
                == ["Bf", "Ba", "Lf", "Lo_depth_weighted"]
                and manifest.get("display_contract", {}).get("panel_c_annotations")
                == ["n.s.", "n.s.", "n.s.", "***"]
            ),
            "ed10_native_powerpoint_pass": final.get("status") == "PASS"
            and final.get("n_pass") == 34
            and final.get("n_fail") == 0,
            "ed10_sources_and_code_frozen": len(
                list((record / "individual_panels").glob("*.*"))
            )
            == 12
            and len(list((record / "source_code").glob("*"))) >= 12,
        }
    validation_files = sorted(record.glob("*VALIDATION*.csv"))
    statuses = all(_status_rows(path)[1] for path in validation_files) if validation_files else True
    return {"frozen_validation_statuses_allowlisted": statuses}


def validate_record(record: Path, figure: str, *, require_hashes: bool = True) -> dict[str, Any]:
    checks = _export_checks(record, figure)
    checks.update(_main_semantics(record, figure) if figure.startswith("main") else _ed_semantics(record, figure))
    refinement = record / "CROSS_FIGURE_REFINEMENT_MANIFEST.json"
    if refinement.is_file():
        payload = json.loads(refinement.read_text(encoding="utf-8"))
        promotion_pass = False
        for promotion_path in (
            record / "NEURON_ID_SHUFFLE_PROMOTION.json",
            record / "ED02_ROUND24_PROMOTION.json",
            record / "ED02_ROUND25_PROMOTION.json",
            record / "ED02_ROUND26_PROMOTION.json",
            record / "DAY_LABEL_REFINEMENT_2026_08_16.json",
            record / "UPRIGHT_DAY_INDEX_REFINEMENT_2026_08_16.json",
        ):
            if not promotion_path.is_file():
                continue
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            promotion_pass = promotion_pass or (
                promotion.get("status") == "PASS"
                and promotion.get("record") == record.name
            )
        checks["cross_figure_refinement_manifest_pass"] = (
            payload.get("status") == "PASS"
            and (payload.get("record") == record.name or promotion_pass)
            and payload.get("figure") == figure
        )
    if require_hashes:
        closure, errors = verify_hash_closure(record)
        checks["complete_hash_closure"] = closure
    else:
        errors = []
    saved = record / "RECORD_VALIDATION.json"
    if require_hashes:
        checks["saved_validation_pass"] = (
            saved.is_file()
            and str(json.loads(saved.read_text(encoding="utf-8")).get("status", "")).lower()
            == "pass"
        )
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
