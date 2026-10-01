#!/usr/bin/env python3
"""Independent read-only audit of the immutable nine-panel ED2 record.

This validator deliberately does not import the promoter.  It prints a JSON
verdict and never writes into the record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
PANEL_ROLES = OrderedDict(
    (
        ("a", "four branch-policy definitions"),
        ("b", "Naive versus Return policy probabilities"),
        ("c", "Return versus Second-day policy probabilities"),
        ("d", "one-policy-at-a-time contribution test"),
        ("e", "Return and Second-day shifts from the Naive policy means"),
        ("f", "four-policy walk calibration"),
        ("g", "depth-dependent P_BS profile and contrasts"),
        ("h", "four-policy versus depth-varying reconstruction"),
        ("i", "depth-varying walk calibration"),
    )
)
PANEL_STEMS = {
    "a": "ed02a_four_policy_schematic_fixed_30min_slide",
    "b": "ed02b_naive_vs_return_policy_scatter_fixed_30min_slide",
    "c": "ed02c_return_vs_second_day_policy_scatter_fixed_30min_slide",
    "d": "ed02d_four_policy_contributions_fixed_30min_slide",
    "e": "ed02e_shift_from_naive_fixed_30min_slide",
    "f": "ed02f_four_policy_calibration_fixed_30min_slide",
    "g": "ed02g_depth_pbs_profile_fixed_30min_slide",
    "h": "ed02h_four_vs_depth_reconstruction_fixed_30min_slide",
    "i": "ed02i_depth_calibration_fixed_30min_slide",
}
FORMATS = {"svg", "pdf", "png", "tiff"}
EXPECTED_TABLES = {
    "panel_a_schematic_rows.csv",
    "panel_b_contrasts.csv",
    "panel_b_session_points.csv",
    "panel_c_contrasts.csv",
    "panel_c_session_points.csv",
    "panel_d_policy_contributions.csv",
    "panel_e_shift_from_naive.csv",
    "panel_f_calibration_summary.csv",
    "panel_f_session_points.csv",
    "panel_g_depth_contrasts.csv",
    "panel_g_group_profiles.csv",
    "panel_g_supported_session_rows.csv",
    "panel_h_four_vs_depth_gap.csv",
    "panel_i_calibration_summary.csv",
    "panel_i_session_points.csv",
}
INPUT_HASHES = {
    "natural/equal_junction_calibration_summary.csv": "77f45c96b7f055b94e363492791ea299c68eb8ebcdb10d2fa0d34bdc4d78a089",
    "natural/equal_junction_depth_group_cluster_stats.csv": "4a86ddff6b85a03ebc0d21a24851848d98203fd4f6cef5a973d93cf1041789d4",
    "natural/window_exposure_audit.csv": "58da24dd41c8d82740420a72f710e9916a4b463d79a21ece63b94b184ce6be82",
    "natural/predictor_outcome_overlap_group_summary.csv": "4e74efc2bcd1fb62427f30ba26f5f9e9de0f9ee2006085cc6ce43e49c5f95c5f",
    "natural/equal_junction_policy_group_cluster_stats.csv": "a1e0763258ae5e1b20887a23ebc722402437f04142737cc3b91760578addf51a",
    "natural/fixed_time_policy_rows.csv": "6a6cd70fe95d45bd2196632acf8001275bc17b537aff3d111aeb3840fc1917bf",
    "natural/equal_junction_simulation_session_rows.csv": "5535b8c62d3f41eed82dc4351f999c54237aeb85500374d1bacf4d1f7a5a160f",
    "natural/estimator_support_comparison.csv": "78001d159800865541b234e83f0074f427a9aa5d76d049e2bf0f511abac50273",
    "strict/primary_30min_strict_joint_policy_contribution_summary.csv": "832f6508705b552e5e9952052e79fd42e0cb8778ae89b0510af933d6af568817",
    "strict/primary_30min_strict_joint_gap_recapitulation_summary.csv": "1a46cceb44c4f0e1c3a60595b42d244924a54b2f76129bb785b93f192d02a038",
    "strict/primary_30min_strict_joint_policy_cluster_stats.csv": "bd4cf4f9b31f62b71a8778c2419aac3602f00b481a0388bba3627953ab3d251b",
    "strict/primary_30min_strict_joint_equal_junction_policy_rows.csv": "6b4244becfc66ee2647b2611e8abc7eea7c8a7efc8013d44a946eb095cbfc6fb",
    "strict/primary_30min_strict_joint_gap_robustness.csv": "59b17c2b22d936c088038217ae4dfc8caa2032725eaabf3014825145ed94c720",
}
PROVENANCE_HASHES = {
    "natural/manifest.json": "1a43cae29d6ef9891240280f983e35e77362f924cd1e0e1f4c9ba73834d6c447",
    "natural/QA.md": "cd2e18205ad3390ae060112b5ca42d0c724388a9ff617061bbf286f9015d7548",
    "natural/REPORT.md": "0bf5a0e13b2e320f1fe8580f98b036dafb057d9054b7d116ee3c6faa0d4a56ec",
    "strict/manifest.json": "e6b43909684042bf95e6348469483e32dbdd3a36a9aa71cd35a95d433796de60",
    "strict/QA.md": "2596ee7fcda0ad45ec64111bb18078556a7e41b32df919c12b1206f1c210ad56",
    "strict/REPORT.md": "b9278d99deec5038c396b95b63d252cfeaea2782dfb35b7cc5f6e51dfc757b43",
}
FROZEN_SOURCE_HASHES = {
    "render_ed03_fixed_time_30min_full_story.py": "1dcdbfa435d9d73d72749f2d18940dadfb195a2681deda6d958143851219d810",
    "analyze_ed03_turning_bias_fixed30min_promoted.py": "851eafbe742e406c811e7af289a9a81f988491e22d0b90de56fb67775fdcb709",
    "analyze_ed03_turning_bias_fixed_time_20_30min.py": "5d7dcc92e61d0ff0f7e46cfe03c6a16767abcc521f3b65db7241b385c538d863",
    "behavior_figure_colors.py": "0bd8386cd3454530faf4d9394bdc6c652fb07db14eb35931c8df695cf70c0bdb",
    "figure_panels.py": "4cedac15faa8f01c68d8059eb9f6cad036337c97e0fbf99f7dfb0c3c19a7c7f4",
}
EXPECTED_PANEL_E = {
    "Bf": (0.012827055087357375, 0.007617753862438459),
    "Ba": (0.11355705377770908, 0.06325800955043082),
    "Lf": (0.08337614205796928, 0.04335588615001362),
    "Lo": (0.09240033737598352, 0.11605492309754195),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float | str, expected: float, atol: float = 5e-9) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol)


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append(
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    def result(self, record_root: Path) -> dict[str, Any]:
        failures = [item for item in self.checks if item["status"] == "fail"]
        return {
            "status": "pass" if not failures else "fail",
            "record": str(record_root),
            "read_only": True,
            "check_count": len(self.checks),
            "failure_count": len(failures),
            "checks": self.checks,
        }


def group_counts(data: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for group in sorted({item["group"] for item in data}):
        selected = [item for item in data if item["group"] == group]
        result[group] = {
            "sessions": len(selected),
            "mice": len({item["animal_id"] for item in selected}),
        }
    return result


def validate_hash_closure(record_root: Path, audit: Audit) -> None:
    ledger_path = record_root / "RECORD_HASHES.csv"
    if not ledger_path.is_file():
        audit.check("hash_ledger_present", False, str(ledger_path))
        return
    ledger = rows(ledger_path)
    ledger_paths = {item["path"] for item in ledger}
    actual_paths = {
        path.relative_to(record_root).as_posix()
        for path in record_root.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    errors = []
    for item in ledger:
        target = record_root / item["path"]
        if not target.is_file():
            errors.append(f"missing:{item['path']}")
        elif sha256(target) != item["sha256"] or target.stat().st_size != int(item["bytes"]):
            errors.append(f"hash_or_bytes:{item['path']}")
    audit.check(
        "complete_fail_closed_hash_inventory",
        ledger_paths == actual_paths and not errors,
        {
            "rows": len(ledger),
            "actual_files_excluding_ledger": len(actual_paths),
            "missing_from_ledger": sorted(actual_paths - ledger_paths),
            "ledger_only": sorted(ledger_paths - actual_paths),
            "errors": errors,
        },
    )


def validate_inventory(record_root: Path, audit: Audit) -> None:
    panel_manifest = rows(record_root / "PANEL_MANIFEST.csv")
    observed_order = list(dict.fromkeys(item["panel"] for item in panel_manifest))
    counts = Counter(item["panel"] for item in panel_manifest)
    roles = {
        panel: {item["role"] for item in panel_manifest if item["panel"] == panel}
        for panel in observed_order
    }
    format_map = {
        panel: {item["format"] for item in panel_manifest if item["panel"] == panel}
        for panel in observed_order
    }
    audit.check(
        "powerpoint_defined_inventory_is_exactly_a_to_i",
        observed_order == list(PANEL_ROLES)
        and len(panel_manifest) == 36
        and set(counts) == set(PANEL_ROLES)
        and all(counts[panel] == 4 for panel in PANEL_ROLES)
        and all(roles[panel] == {PANEL_ROLES[panel]} for panel in PANEL_ROLES)
        and all(format_map[panel] == FORMATS for panel in PANEL_ROLES),
        {
            "order": observed_order,
            "counts": dict(counts),
            "roles": {panel: sorted(values) for panel, values in roles.items()},
        },
    )
    artifact_errors = []
    raster_errors = []
    svg_errors = []
    pdf_errors = []
    for item in panel_manifest:
        target = record_root / item["path"]
        expected_stem = PANEL_STEMS.get(item["panel"])
        if (
            not target.is_file()
            or sha256(target) != item["sha256"]
            or target.stat().st_size != int(item["bytes"])
            or target.stem != expected_stem
        ):
            artifact_errors.append(item["path"])
            continue
        width = float(item["width_in"])
        height = float(item["height_in"])
        if item["format"] in {"png", "tiff"}:
            with Image.open(target) as image:
                if image.size != (round(width * 600), round(height * 600)):
                    raster_errors.append(
                        {"path": item["path"], "size": image.size, "expected": [round(width * 600), round(height * 600)]}
                    )
        elif item["format"] == "svg":
            if "<text" not in target.read_text(encoding="utf-8").lower():
                svg_errors.append(item["path"])
        elif item["format"] == "pdf":
            with fitz.open(target) as document:
                rect = document[0].rect
                if document.page_count != 1 or not (
                    math.isclose(rect.width / 72, width, abs_tol=0.015)
                    and math.isclose(rect.height / 72, height, abs_tol=0.015)
                ):
                    pdf_errors.append(item["path"])
    audit.check("panel_manifest_hash_size_stem_parity", not artifact_errors, artifact_errors)
    audit.check("panel_rasters_are_600dpi_final_size", not raster_errors, raster_errors)
    audit.check("panel_svg_text_remains_editable", not svg_errors, svg_errors)
    audit.check("panel_pdfs_are_single_page_final_size", not pdf_errors, pdf_errors)

    forbidden = [
        path.relative_to(record_root).as_posix()
        for path in record_root.rglob("*")
        if path.is_file() and ("ed02j" in path.name.lower() or "panel_j" in path.name.lower())
    ]
    audit.check("no_panel_j_payload_or_role", not forbidden, forbidden)


def validate_tables_and_science(record_root: Path, audit: Audit) -> None:
    manifest = rows(record_root / "TABLE_MANIFEST.csv")
    observed = {Path(item["path"]).name for item in manifest}
    errors = []
    window_errors = []
    for item in manifest:
        target = record_root / item["path"]
        if (
            not target.is_file()
            or sha256(target) != item["sha256"]
            or target.stat().st_size != int(item["bytes"])
        ):
            errors.append(item["path"])
            continue
        data = rows(target)
        if len(data) != int(item["rows"]):
            errors.append(f"rows:{item['path']}")
        for row_index, row in enumerate(data, start=2):
            if "window_minutes" not in row or not close(row["window_minutes"], 30):
                window_errors.append(f"{item['path']}:{row_index}")
    audit.check(
        "plot_ready_inventory_is_exactly_15_tables",
        len(manifest) == 15 and observed == EXPECTED_TABLES and not errors,
        {"observed": sorted(observed), "errors": errors},
    )
    audit.check("all_plot_ready_rows_declare_30min", not window_errors, window_errors)

    a = rows(record_root / "plot_ready/panel_a_schematic_rows.csv")
    a_map = {item["row_key"]: item["policy_label"] for item in a}
    audit.check(
        "panel_a_canonical_schematic_helper_and_mapping",
        a_map
        == {
            "stem_forward": "$P_{SF}$",
            "stem_to_branch": "$P_{SA}$",
            "branch_forward": "$P_{BF}$",
            "branch_to_stem": "$P_{BS}$",
        }
        and all(item["helper"] == "render_grid_route_choice_schematic" for item in a),
        a_map,
    )

    b_counts = group_counts(rows(record_root / "plot_ready/panel_b_session_points.csv"))
    c_counts = group_counts(rows(record_root / "plot_ready/panel_c_session_points.csv"))
    audit.check(
        "panels_b_c_current_support",
        b_counts == {"LTM": {"sessions": 58, "mice": 18}, "naive": {"sessions": 22, "mice": 22}}
        and c_counts == {"LTM": {"sessions": 58, "mice": 18}, "experienced": {"sessions": 74, "mice": 24}}
        and len(rows(record_root / "plot_ready/panel_b_contrasts.csv")) == 4
        and len(rows(record_root / "plot_ready/panel_c_contrasts.csv")) == 4,
        {"b": b_counts, "c": c_counts},
    )

    d = {
        item["parameter"]: round(100 * float(item["delta_efficiency_mean"]), 1)
        for item in rows(record_root / "plot_ready/panel_d_policy_contributions.csv")
    }
    audit.check(
        "panel_d_current_policy_contributions",
        d == {"Bf": -0.4, "Ba": -2.7, "Lf": 8.0, "Lo": 6.9},
        d,
    )

    e_rows = rows(record_root / "plot_ready/panel_e_shift_from_naive.csv")
    e_map = {
        item["metric"]: (
            float(item["return_minus_naive"]),
            float(item["second_day_minus_naive"]),
        )
        for item in e_rows
    }
    e_ok = (
        set(e_map) == set(EXPECTED_PANEL_E)
        and all(
            close(e_map[key][0], expected[0], 5e-12)
            and close(e_map[key][1], expected[1], 5e-12)
            for key, expected in EXPECTED_PANEL_E.items()
        )
        and all(
            int(item["n_naive_sessions"]) == 23
            and int(item["n_return_sessions"]) == 58
            and int(item["n_second_day_sessions"]) == 74
            and item["support"] == "natural primary-policy support"
            and item["point_weighting"] == "session-weighted group means"
            for item in e_rows
        )
    )
    audit.check("panel_e_current_fixed30_natural_values", e_ok, e_map)

    natural_rows = [
        item
        for item in rows(record_root / "analysis_inputs/natural/fixed_time_policy_rows.csv")
        if close(item["window_minutes"], 30) and as_bool(item["primary_policy_support"])
    ]
    natural_counts = Counter(item["group"] for item in natural_rows)
    recomputed: dict[str, tuple[float, float]] = {}
    for metric in EXPECTED_PANEL_E:
        means = {
            group: statistics.fmean(
                float(item[metric]) for item in natural_rows if item["group"] == group
            )
            for group in ("naive", "LTM", "experienced")
        }
        recomputed[metric] = (
            means["LTM"] - means["naive"],
            means["experienced"] - means["naive"],
        )
    audit.check(
        "panel_e_recomputes_from_frozen_natural_rows",
        natural_counts == {"naive": 23, "LTM": 58, "experienced": 74}
        and all(
            close(e_map[key][0], recomputed[key][0], 5e-12)
            and close(e_map[key][1], recomputed[key][1], 5e-12)
            for key in EXPECTED_PANEL_E
        ),
        {"counts": dict(natural_counts), "recomputed": recomputed},
    )

    f_contract = {
        item["group_label"]: (
            int(item["n_session_rows"]),
            int(item["n_mice"]),
            as_bool(item["combined_gate_pass"]),
        )
        for item in rows(record_root / "plot_ready/panel_f_calibration_summary.csv")
    }
    audit.check(
        "panel_f_current_four_policy_calibration",
        f_contract
        == {"Naive": (23, 23, False), "Return": (58, 18, False), "Second day": (71, 24, False)},
        f_contract,
    )

    g_counts = group_counts(rows(record_root / "plot_ready/panel_g_supported_session_rows.csv"))
    audit.check(
        "panel_g_current_depth_support_and_contrasts",
        g_counts
        == {
            "LTM": {"sessions": 58, "mice": 18},
            "experienced": {"sessions": 74, "mice": 24},
            "naive": {"sessions": 22, "mice": 22},
        }
        and len(rows(record_root / "plot_ready/panel_g_group_profiles.csv")) == 9
        and len(rows(record_root / "plot_ready/panel_g_depth_contrasts.csv")) == 6,
        g_counts,
    )

    h = {
        item["model"]: item
        for item in rows(record_root / "plot_ready/panel_h_four_vs_depth_gap.csv")
    }
    h_values = (
        round(100 * float(h["four_bias"]["fraction_recapitulated"]), 1),
        round(100 * float(h["depth_stratified"]["fraction_recapitulated"]), 1),
        round(100 * float(h["depth_stratified"]["fraction_ci_low"]), 1),
        round(100 * float(h["depth_stratified"]["fraction_ci_high"]), 1),
    )
    audit.check("panel_h_current_reconstruction", h_values == (53.3, 99.5, 84.1, 122.7), h_values)

    i_contract = {
        item["group_label"]: (
            int(item["n_session_rows"]),
            int(item["n_mice"]),
            as_bool(item["combined_gate_pass"]),
        )
        for item in rows(record_root / "plot_ready/panel_i_calibration_summary.csv")
    }
    audit.check(
        "panel_i_current_depth_calibration",
        i_contract
        == {"Naive": (23, 23, True), "Return": (58, 18, True), "Second day": (71, 24, False)},
        i_contract,
    )


def validate_inputs_and_code(record_root: Path, audit: Audit) -> None:
    manifest = rows(record_root / "INPUT_MANIFEST.csv")
    declared = {item["record_path"].replace("analysis_inputs/", "", 1): item for item in manifest}
    errors = []
    forbidden_tokens = ("neuron", "cimap", "footprint", "manual_mask", "manual_exclusion")
    forbidden_hits = []
    for relative, expected_hash in INPUT_HASHES.items():
        item = declared.get(relative)
        target = record_root / "analysis_inputs" / relative
        if (
            item is None
            or item["sha256"] != expected_hash
            or not target.is_file()
            or sha256(target) != expected_hash
        ):
            errors.append(relative)
            continue
        header = next(csv.reader(target.open("r", encoding="utf-8-sig", newline="")), [])
        hits = [column for column in header if any(token in column.lower() for token in forbidden_tokens)]
        if hits:
            forbidden_hits.append({"path": relative, "columns": hits})
    audit.check(
        "exact_frozen_current_analysis_inputs_13",
        len(manifest) == 13 and set(declared) == set(INPUT_HASHES) and not errors,
        errors,
    )
    audit.check(
        "behavior_only_manual_masks_not_applicable",
        not forbidden_hits,
        forbidden_hits,
    )

    provenance_errors = []
    for relative, expected_hash in PROVENANCE_HASHES.items():
        target = record_root / "analysis_provenance" / relative
        if not target.is_file() or sha256(target) != expected_hash:
            provenance_errors.append(relative)
    audit.check("exact_analysis_provenance_six_files", not provenance_errors, provenance_errors)

    code_files = {path.name: path for path in (record_root / "source_code").glob("*.py")}
    expected_code_names = {
        *FROZEN_SOURCE_HASHES,
        "render_ed02_slide3_powerpoint_inventory.py",
        "promote_ed02_slide_inventory_version_record.py",
        "validate_ed02_slide_inventory_version_record.py",
    }
    code_errors = []
    for name, expected_hash in FROZEN_SOURCE_HASHES.items():
        target = code_files.get(name)
        if target is None or sha256(target) != expected_hash:
            code_errors.append(name)
    audit.check(
        "necessary_source_code_frozen_without_pycache",
        set(code_files) == expected_code_names
        and not code_errors
        and not list((record_root / "source_code").rglob("*.pyc")),
        {"observed": sorted(code_files), "hash_errors": code_errors},
    )


def _rect_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    dx = min(left["x_in"] + left["width_in"], right["x_in"] + right["width_in"]) - max(
        left["x_in"], right["x_in"]
    )
    dy = min(left["y_in"] + left["height_in"], right["y_in"] + right["height_in"]) - max(
        left["y_in"], right["y_in"]
    )
    return dx > 1e-9 and dy > 1e-9


def validate_composite(record_root: Path, audit: Audit) -> None:
    manifest = json.loads(
        (record_root / "ED02_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8")
    )
    placements = manifest.get("placements", [])
    placement_order = [item["panel"] for item in placements]
    bounds_errors = [
        item["panel"]
        for item in placements
        if item["x_in"] < 0
        or item["y_in"] < 0
        or item["x_in"] + item["width_in"] > 7.5 + 1e-9
        or item["y_in"] + item["height_in"] > 7.5 + 1e-9
    ]
    overlaps = []
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            if _rect_overlap(left, right):
                overlaps.append(f"{left['panel']}-{right['panel']}")
    input_errors = []
    for item in placements:
        target = record_root / item["input_path"]
        if not target.is_file() or sha256(target) != item["input_sha256"]:
            input_errors.append(item["panel"])
    audit.check(
        "composite_matches_three_by_three_slide_order",
        manifest.get("panel_order") == list(PANEL_ROLES)
        and manifest.get("row_order") == [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        and placement_order == list(PANEL_ROLES)
        and [(item["row"], item["column"]) for item in placements]
        == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
        {"order": placement_order, "rows": manifest.get("row_order")},
    )
    audit.check(
        "composite_panels_in_bounds_nonoverlapping_and_hash_exact",
        not bounds_errors and not overlaps and not input_errors,
        {"bounds": bounds_errors, "overlaps": overlaps, "input_hash_errors": input_errors},
    )
    output_errors = []
    for item in manifest.get("outputs", {}).values():
        target = record_root / item["path"]
        if (
            not target.is_file()
            or sha256(target) != item["sha256"]
            or target.stat().st_size != int(item["bytes"])
        ):
            output_errors.append(item.get("path"))
    pdf_path = record_root / "ed02_slide_inventory_version_record.pdf"
    with fitz.open(pdf_path) as document:
        pdf_contract = (
            document.page_count == 1
            and math.isclose(document[0].rect.width, 540, abs_tol=0.1)
            and math.isclose(document[0].rect.height, 540, abs_tol=0.1)
        )
        label_spans = [
            span
            for block in document[0].get_text("dict").get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text") in PANEL_ROLES
            and math.isclose(float(span.get("size", 0)), 10.5, abs_tol=0.05)
            and "Arial" in str(span.get("font", ""))
        ]
    with Image.open(record_root / "ed02_slide_inventory_version_record_600dpi.png") as image:
        png_contract = image.size == (4500, 4500)
    with Image.open(record_root / "ed02_slide_inventory_version_record_600dpi.tiff") as image:
        tiff_contract = image.size == (4500, 4500)
    audit.check(
        "composite_one_page_four_format_export_contract",
        not output_errors
        and set(manifest.get("outputs", {})) == {"pdf", "svg", "png", "tiff"}
        and pdf_contract
        and png_contract
        and tiff_contract,
        {"output_errors": output_errors, "pdf": pdf_contract, "png": png_contract, "tiff": tiff_contract},
    )
    label_contract = manifest.get("panel_label", {})
    audit.check(
        "composite_panel_letters_are_arial_bold_10p5",
        label_contract.get("font") == "Arial Bold"
        and label_contract.get("font_file") == "arialbd.ttf"
        and close(label_contract.get("size_pt", 0), 10.5, 0.05)
        and {span["text"] for span in label_spans} == set(PANEL_ROLES),
        {
            "manifest": label_contract,
            "pdf_spans": [
                {"text": span["text"], "font": span.get("font"), "size": span.get("size")}
                for span in label_spans
            ],
        },
    )


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _all_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _all_strings(nested)]
    return []


def validate_portability(record_root: Path, audit: Audit) -> None:
    authority_json = (
        "SOURCE_MANIFEST.json",
        "RENDER_RESULT.json",
        "ED02_COMPOSITE_MANIFEST.json",
        "ED02_VERSION_RECORD_MANIFEST.json",
        "RECORD_VALIDATION.json",
    )
    transient_hits = []
    for name in authority_json:
        payload = json.loads((record_root / name).read_text(encoding="utf-8"))
        for value in _all_strings(payload):
            normalized = value.replace("\\", "/")
            if (
                (len(normalized) >= 3 and normalized[1:3] == ":/")
                or normalized.lower().startswith("tmp/")
                or "/tmp/" in normalized.lower()
                or ".staging-" in normalized.lower()
            ):
                transient_hits.append({"manifest": name, "value": value})
    path_hits = []
    for name, column in (
        ("PANEL_MANIFEST.csv", "path"),
        ("TABLE_MANIFEST.csv", "path"),
        ("INPUT_MANIFEST.csv", "record_path"),
    ):
        for item in rows(record_root / name):
            value = item[column].replace("\\", "/")
            if (
                Path(value).is_absolute()
                or value.lower().startswith("tmp/")
                or "/tmp/" in value.lower()
                or ".staging-" in value.lower()
            ):
                path_hits.append({"manifest": name, "value": value})
    render_result = json.loads((record_root / "RENDER_RESULT.json").read_text(encoding="utf-8"))
    audit.check(
        "authority_manifests_are_record_local_and_portable",
        not transient_hits
        and not path_hits
        and render_result.get("output_root") == "record-local copied render payload",
        {
            "transient_json_hits": transient_hits,
            "transient_path_hits": path_hits,
            "render_output_root": render_result.get("output_root"),
        },
    )
    promotion = json.loads((record_root / "AUDIT_PROMOTION.json").read_text(encoding="utf-8"))
    promotion_transient = [
        item
        for item in promotion
        if "tmp/" in str(item.get("source", "")).replace("\\", "/").lower()
        or ".staging-" in str(item.get("source", "")).lower()
        or ".staging-" in str(item.get("record_path", "")).lower()
    ]
    audit.check(
        "promotion_audit_uses_portable_renderer_output_labels",
        not promotion_transient
        and any(str(item.get("source", "")).startswith("renderer_output/") for item in promotion),
        promotion_transient,
    )
    upstream = rows(record_root / "UPSTREAM_INPUT_MANIFEST.csv")
    audit.check(
        "historical_absolute_upstream_input_provenance_retained",
        len(upstream) == 13
        and all(Path(item["path"]).is_absolute() for item in upstream)
        and all(item["sha256"] in set(INPUT_HASHES.values()) for item in upstream),
        {"rows": len(upstream), "absolute_paths": sum(Path(item["path"]).is_absolute() for item in upstream)},
    )


def validate_saved_metadata(record_root: Path, audit: Audit) -> None:
    saved = json.loads((record_root / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    version = json.loads(
        (record_root / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit.check(
        "saved_promotion_verdict_pass",
        saved.get("status") == "pass" and saved.get("failure_count") == 0,
        {"status": saved.get("status"), "failure_count": saved.get("failure_count")},
    )
    audit.check(
        "promotion_complete_precedes_hash_contract",
        (record_root / "PROMOTION_COMPLETE.txt").is_file()
        and "RECORD_VALIDATION.json and PROMOTION_COMPLETE.txt are written before"
        in saved.get("hash_policy", ""),
        (record_root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8")
        if (record_root / "PROMOTION_COMPLETE.txt").is_file()
        else "missing",
    )
    audit.check(
        "version_manifest_records_slide_authority_and_mask_scope",
        version.get("panel_order") == list(PANEL_ROLES)
        and version.get("row_order") == [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        and version.get("panel_inventory_authority") == "supplied PowerPoint slide 3"
        and version.get("record") == record_root.name
        and ".staging-" not in json.dumps(version)
        and version.get("manual_mask_applicability") == "not applicable; behavior-only analysis"
        and version.get("export_contract", {}).get("individual_panels") == 9
        and version.get("export_contract", {}).get("plot_ready_tables") == 15
        and version.get("export_contract", {}).get("renderer_input_tables") == 13,
        {
            "panel_order": version.get("panel_order"),
            "authority": version.get("panel_inventory_authority"),
            "record": version.get("record"),
            "export": version.get("export_contract"),
        },
    )


def validate(record_root: Path) -> dict[str, Any]:
    audit = Audit()
    if not record_root.is_dir():
        audit.check("record_exists", False, str(record_root))
        return audit.result(record_root)
    required = {
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "INPUT_MANIFEST.csv",
        "SOURCE_MANIFEST.json",
        "RENDER_RESULT.json",
        "ED02_COMPOSITE_MANIFEST.json",
        "ED02_VERSION_RECORD_MANIFEST.json",
        "RECORD_VALIDATION.json",
        "PROMOTION_COMPLETE.txt",
        "RECORD_HASHES.csv",
    }
    missing = sorted(name for name in required if not (record_root / name).is_file())
    audit.check("required_record_authority_files_present", not missing, missing)
    if missing:
        return audit.result(record_root)
    validate_hash_closure(record_root, audit)
    validate_inventory(record_root, audit)
    validate_tables_and_science(record_root, audit)
    validate_inputs_and_code(record_root, audit)
    validate_composite(record_root, audit)
    validate_portability(record_root, audit)
    validate_saved_metadata(record_root, audit)
    return audit.result(record_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(args.record_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
