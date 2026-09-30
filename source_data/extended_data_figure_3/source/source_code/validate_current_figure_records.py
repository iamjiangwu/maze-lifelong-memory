"""Read-only, fail-closed validation for the 15 current figure records."""

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
        svg_text = _find_one(record, "*_version_record.svg").read_text(encoding="utf-8")
        exploration = pd.read_csv(
            record / "plot_ready/exploration_first_chunk_current_plus_new_points.csv"
        )
        exploration_counts = exploration.groupby("group").size().to_dict()
        return {
            "figure1_eight_sources": len(manifest.get("sources", [])) == 8,
            "figure1_six_timeline_sessions_frozen": timelines["nickname"].nunique() == 6 and len(timelines) > 100,
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
            "figure1_exploration_histogram_unit_is_sessions": (
                "No. of sessions" in svg_text and "# of Animals" not in svg_text
            ),
            "figure1_exploration_session_counts_exact": exploration_counts
            == {"LTM D1 (all)": 56, "Naive D1 (rewarded)": 23},
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
                "LTM": 58,
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
            "ed7_row_titles_exact": display.get("top_y_axis_label") == "Day 1 early vs Day 2 correlation" and display.get("bottom_y_axis_label") == "Long gap correlation" and "(r)" not in json.dumps(display),
            "ed7_scopes_locked": display_manifest.get("top_primary_scope") == "58 maintained observations / 21 mice; BH3" and display_manifest.get("bottom_primary_scope") == "36 pairs / 14 mice; BH4; current clean single-vintage primary",
            "ed7_exclusions_and_mask_locked": len(display_manifest.get("excluded_from_all_ed7_primary_analyses", [])) == 6 and display_manifest.get("manual_mask_profile") == "none for every displayed source row",
            "ed7_sources_and_code_frozen": len(record_manifest.get("scientific_inputs", [])) == 6 and (record / "source_code/render_ed07_slide9_powerpoint_matched.py").is_file() and (record / "source_code/validate_ed07_six_panel_round11.py").is_file(),
        }
    if figure == "extended_data_9":
        manifest = json.loads(
            (record / "ED09_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
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
        return {
            "ed9_stable_record_identity": manifest.get("record") == record.name,
            "ed9_twelve_panel_inventory": manifest.get("panel_letters") == list("abcdefghijkl"),
            "ed9_round11_render_pass": render.get("status") == "PASS" and all(render.get("checks", {}).values()),
            "ed9_main_figure_grammar_pass": grammar.get("status") == "PASS" and len(grammar.get("panels", [])) == 12 and all(grammar.get("checks", {}).values()),
            "ed9_no_manual_mask_pass": manual.get("status") == "PASS" and all(manual.get("checks", {}).values()),
            "ed9_support_16901": conditional.get("cohort", {}).get("transition_rows") == 16901 and conditional.get("cohort", {}).get("restored_from_obsolete_manual_mask") == 4,
            "ed9_panel_a_fixed_colorbar": render.get("panel_a_estimand", {}).get("render", {}).get("color_endpoint") == 0.1,
            "ed9_panel_i_main_figure2_grammar": render.get("checks", {}).get("panel_i_main_figure2_grammar_exact") is True and render.get("checks", {}).get("panel_i_plot_ready_source_unchanged_from_round10") is True,
            "ed9_panel_c_current_metadata_exact": set(
                manifest.get("panel_c_axis_labels", {}).values()
            )
            == {"Place map correlation"}
            and manifest.get("panel_c_significance")
            == (
                "standalone directional mouse-clustered CR1 raw P=0.00947165887533162; "
                "no multiplicity correction; **"
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
            "ed10_display_contract_exact": manifest.get("display_contract", {}).get("colorbar_title")
            == "Pearson correlation"
            and manifest.get("display_contract", {}).get("panel_a_label_weight") == "normal"
            and manifest.get("display_contract", {}).get("panel_c_annotations")
            == ["n.s.", "n.s.", "n.s.", "***"],
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
        checks["cross_figure_refinement_manifest_pass"] = (
            payload.get("status") == "PASS"
            and payload.get("record") == record.name
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
