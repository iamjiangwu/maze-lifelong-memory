#!/usr/bin/env python3
"""Promote the PowerPoint-defined nine-panel ED2 inventory immutably.

The source PowerPoint fixes the scientific panel inventory and order at a--i.
This promoter accepts only the final Python-rendered slide panels, freezes the
13 current fixed-30-minute behavioral analysis inputs and their provenance,
builds a one-page 3 x 3 review composite, writes the saved verdict and
completion stamp, and only then hashes every record file except the hash
ledger itself.

An existing target is never modified.  Use
``validate_ed02_slide_inventory_version_record.py`` for later read-only audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "tmp" / "ed02_slide3_inventory_correction_2026-08-08" / "panels"
)
CURRENT_SOURCE_RECORD = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_full_audit"
)
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
FORMATS = ("svg", "pdf", "png", "tiff")
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

# These hashes identify the exact maintained analysis inputs used by the
# accepted fixed-30-minute render.  They intentionally do not follow mutable
# live paths during promotion.
ANALYSIS_INPUTS = OrderedDict(
    (
        (
            "natural_calibration",
            (
                "natural/equal_junction_calibration_summary.csv",
                "77f45c96b7f055b94e363492791ea299c68eb8ebcdb10d2fa0d34bdc4d78a089",
            ),
        ),
        (
            "natural_depth_contrasts",
            (
                "natural/equal_junction_depth_group_cluster_stats.csv",
                "4a86ddff6b85a03ebc0d21a24851848d98203fd4f6cef5a973d93cf1041789d4",
            ),
        ),
        (
            "natural_exposure",
            (
                "natural/window_exposure_audit.csv",
                "58da24dd41c8d82740420a72f710e9916a4b463d79a21ece63b94b184ce6be82",
            ),
        ),
        (
            "natural_outcome_overlap",
            (
                "natural/predictor_outcome_overlap_group_summary.csv",
                "4e74efc2bcd1fb62427f30ba26f5f9e9de0f9ee2006085cc6ce43e49c5f95c5f",
            ),
        ),
        (
            "natural_policy_contrasts",
            (
                "natural/equal_junction_policy_group_cluster_stats.csv",
                "a1e0763258ae5e1b20887a23ebc722402437f04142737cc3b91760578addf51a",
            ),
        ),
        (
            "natural_policy_rows",
            (
                "natural/fixed_time_policy_rows.csv",
                "6a6cd70fe95d45bd2196632acf8001275bc17b537aff3d111aeb3840fc1917bf",
            ),
        ),
        (
            "natural_simulation_rows",
            (
                "natural/equal_junction_simulation_session_rows.csv",
                "5535b8c62d3f41eed82dc4351f999c54237aeb85500374d1bacf4d1f7a5a160f",
            ),
        ),
        (
            "natural_support",
            (
                "natural/estimator_support_comparison.csv",
                "78001d159800865541b234e83f0074f427a9aa5d76d049e2bf0f511abac50273",
            ),
        ),
        (
            "strict_contributions",
            (
                "strict/primary_30min_strict_joint_policy_contribution_summary.csv",
                "832f6508705b552e5e9952052e79fd42e0cb8778ae89b0510af933d6af568817",
            ),
        ),
        (
            "strict_gap",
            (
                "strict/primary_30min_strict_joint_gap_recapitulation_summary.csv",
                "1a46cceb44c4f0e1c3a60595b42d244924a54b2f76129bb785b93f192d02a038",
            ),
        ),
        (
            "strict_policy_contrasts",
            (
                "strict/primary_30min_strict_joint_policy_cluster_stats.csv",
                "bd4cf4f9b31f62b71a8778c2419aac3602f00b481a0388bba3627953ab3d251b",
            ),
        ),
        (
            "strict_policy_rows",
            (
                "strict/primary_30min_strict_joint_equal_junction_policy_rows.csv",
                "6b4244becfc66ee2647b2611e8abc7eea7c8a7efc8013d44a946eb095cbfc6fb",
            ),
        ),
        (
            "strict_robustness",
            (
                "strict/primary_30min_strict_joint_gap_robustness.csv",
                "59b17c2b22d936c088038217ae4dfc8caa2032725eaabf3014825145ed94c720",
            ),
        ),
    )
)
PROVENANCE_FILES = (
    "natural/manifest.json",
    "natural/QA.md",
    "natural/REPORT.md",
    "strict/manifest.json",
    "strict/QA.md",
    "strict/REPORT.md",
)
SOURCE_CODE = (
    REPO_ROOT / "scripts" / "render_ed02_slide3_powerpoint_inventory.py",
    CURRENT_SOURCE_RECORD / "source_code" / "render_ed03_fixed_time_30min_full_story.py",
    CURRENT_SOURCE_RECORD / "source_code" / "analyze_ed03_turning_bias_fixed30min_promoted.py",
    CURRENT_SOURCE_RECORD / "source_code" / "analyze_ed03_turning_bias_fixed_time_20_30min.py",
    CURRENT_SOURCE_RECORD / "source_code" / "behavior_figure_colors.py",
    CURRENT_SOURCE_RECORD / "source_code" / "figure_panels.py",
    Path(__file__),
    REPO_ROOT / "scripts" / "validate_ed02_slide_inventory_version_record.py",
)

EXPECTED_PANEL_E = {
    "Bf": (0.012827055087357375, 0.007617753862438459),
    "Ba": (0.11355705377770908, 0.06325800955043082),
    "Lf": (0.08337614205796928, 0.04335588615001362),
    "Lo": (0.09240033737598352, 0.11605492309754195),
}
ARIAL_BOLD_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fieldnames: Iterable[str], data: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def close(actual: float | str, expected: float, atol: float = 5e-12) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append(
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    def result(self) -> dict[str, Any]:
        failures = [item for item in self.checks if item["status"] == "fail"]
        return {
            "status": "pass" if not failures else "fail",
            "check_count": len(self.checks),
            "failure_count": len(failures),
            "checks": self.checks,
        }


def _verify_manifest_file(root: Path, item: dict[str, str]) -> str | None:
    target = root / item["path"]
    if not target.is_file():
        return f"missing:{item['path']}"
    if sha256(target) != item["sha256"]:
        return f"hash:{item['path']}"
    if target.stat().st_size != int(item["bytes"]):
        return f"bytes:{item['path']}"
    return None


def validate_source(source_root: Path, source_record: Path) -> dict[str, Any]:
    audit = Audit()
    required = {
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "SOURCE_MANIFEST.json",
        "RENDER_RESULT.json",
    }
    missing = sorted(name for name in required if not (source_root / name).is_file())
    audit.check("source_manifests_present", not missing, missing)
    if missing:
        return audit.result()

    panel_manifest = rows(source_root / "PANEL_MANIFEST.csv")
    observed_order = list(dict.fromkeys(item["panel"] for item in panel_manifest))
    role_map = {
        panel: {item["role"] for item in panel_manifest if item["panel"] == panel}
        for panel in observed_order
    }
    counts = Counter(item["panel"] for item in panel_manifest)
    format_map = {
        panel: {item["format"] for item in panel_manifest if item["panel"] == panel}
        for panel in observed_order
    }
    parity_ok = (
        observed_order == list(PANEL_ROLES)
        and len(panel_manifest) == 36
        and all(counts[panel] == 4 for panel in PANEL_ROLES)
        and all(format_map[panel] == set(FORMATS) for panel in PANEL_ROLES)
        and all(role_map[panel] == {PANEL_ROLES[panel]} for panel in PANEL_ROLES)
    )
    audit.check(
        "powerpoint_panel_role_order_parity_a_to_i",
        parity_ok,
        {
            "order": observed_order,
            "counts": dict(counts),
            "roles": {panel: sorted(values) for panel, values in role_map.items()},
        },
    )
    audit.check(
        "no_panel_j_or_extra_role",
        "j" not in counts and set(counts) == set(PANEL_ROLES),
        sorted(counts),
    )

    artifact_errors = [
        error
        for item in panel_manifest
        if (error := _verify_manifest_file(source_root, item)) is not None
    ]
    stems_ok = all(
        Path(item["path"]).stem == PANEL_STEMS[item["panel"]]
        for item in panel_manifest
        if item["panel"] in PANEL_STEMS
    )
    audit.check(
        "panel_artifacts_36_match_manifest",
        not artifact_errors and stems_ok,
        {"errors": artifact_errors, "stems_ok": stems_ok},
    )
    svg_without_text = []
    for panel in PANEL_ROLES:
        path = source_root / "panels" / f"{PANEL_STEMS[panel]}.svg"
        if "<text" not in path.read_text(encoding="utf-8").lower():
            svg_without_text.append(path.name)
    audit.check("editable_svg_text_a_to_i", not svg_without_text, svg_without_text)

    table_manifest = rows(source_root / "TABLE_MANIFEST.csv")
    table_paths = {Path(item["path"]).name for item in table_manifest}
    table_errors = [
        error
        for item in table_manifest
        if (error := _verify_manifest_file(source_root, item)) is not None
    ]
    row_errors = []
    window_errors = []
    for item in table_manifest:
        table_path = source_root / item["path"]
        data = rows(table_path)
        if len(data) != int(item["rows"]):
            row_errors.append(item["path"])
        for index, row in enumerate(data, start=2):
            if "window_minutes" not in row or not close(row["window_minutes"], 30, 1e-9):
                window_errors.append(f"{item['path']}:{index}")
    audit.check(
        "plot_ready_tables_15_exact",
        len(table_manifest) == 15
        and table_paths == EXPECTED_TABLES
        and not table_errors
        and not row_errors,
        {
            "observed": sorted(table_paths),
            "manifest_errors": table_errors,
            "row_errors": row_errors,
        },
    )
    audit.check("all_plot_ready_rows_are_fixed_30min", not window_errors, window_errors)

    render_result = json.loads((source_root / "RENDER_RESULT.json").read_text(encoding="utf-8"))
    audit.check(
        "saved_render_result_passes_nine_panel_contract",
        render_result.get("status") == "pass"
        and render_result.get("panel_count") == 9
        and render_result.get("panel_order") == list(PANEL_ROLES)
        and render_result.get("panel_artifacts") == 36
        and render_result.get("plot_ready_tables") == 15
        and render_result.get("dpi") == 600,
        {
            **{key: value for key, value in render_result.items() if key != "output_root"},
            "output_root": "source output location intentionally omitted from saved validation",
        },
    )

    source_manifest = json.loads((source_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    declared_inputs = {item["key"]: item for item in source_manifest.get("inputs", [])}
    input_errors = []
    for key, (relative, expected_hash) in ANALYSIS_INPUTS.items():
        declaration = declared_inputs.get(key, {})
        record_path = source_record / "analysis_inputs" / relative
        if declaration.get("path") != f"analysis_inputs/{relative}":
            input_errors.append(f"path:{key}")
        if declaration.get("sha256") != expected_hash:
            input_errors.append(f"declared_hash:{key}")
        if not record_path.is_file() or sha256(record_path) != expected_hash:
            input_errors.append(f"record_hash:{key}")
    audit.check(
        "exact_current_analysis_inputs_13",
        len(declared_inputs) == 13 and not input_errors,
        input_errors,
    )
    frozen_renderer = source_record / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
    audit.check(
        "frozen_renderer_matches_source_manifest",
        source_manifest.get("frozen_renderer", {}).get("sha256") == sha256(frozen_renderer),
        source_manifest.get("frozen_renderer", {}),
    )
    audit.check(
        "behavior_only_manual_masks_not_applicable",
        source_manifest.get("manual_mask_applicability")
        == "not applicable; behavior-only analysis",
        source_manifest.get("manual_mask_applicability"),
    )

    e_rows = rows(source_root / "plot_ready" / "panel_e_shift_from_naive.csv")
    e_map = {
        item["metric"]: (
            float(item["return_minus_naive"]),
            float(item["second_day_minus_naive"]),
        )
        for item in e_rows
    }
    e_values_ok = (
        set(e_map) == set(EXPECTED_PANEL_E)
        and all(
            close(e_map[key][0], expected[0]) and close(e_map[key][1], expected[1])
            for key, expected in EXPECTED_PANEL_E.items()
        )
        and all(
            int(item["n_naive_sessions"]) == 23
            and int(item["n_return_sessions"]) == 58
            and int(item["n_second_day_sessions"]) == 74
            and item["support"] == "natural primary-policy support"
            for item in e_rows
        )
    )
    audit.check("panel_e_current_fixed30_natural_support_values", e_values_ok, e_map)

    natural_rows = [
        item
        for item in rows(source_record / "analysis_inputs/natural/fixed_time_policy_rows.csv")
        if close(item["window_minutes"], 30, 1e-9)
        and as_bool(item["primary_policy_support"])
    ]
    group_names = {"naive": "naive", "return": "LTM", "second_day": "experienced"}
    recomputed: dict[str, tuple[float, float]] = {}
    for metric in EXPECTED_PANEL_E:
        means = {
            label: statistics.fmean(
                float(item[metric]) for item in natural_rows if item["group"] == group
            )
            for label, group in group_names.items()
        }
        recomputed[metric] = (
            means["return"] - means["naive"],
            means["second_day"] - means["naive"],
        )
    audit.check(
        "panel_e_recomputed_from_frozen_natural_rows",
        all(
            close(e_map[key][0], recomputed[key][0])
            and close(e_map[key][1], recomputed[key][1])
            for key in EXPECTED_PANEL_E
        ),
        recomputed,
    )
    return audit.result()


def copy_file(
    source: Path,
    destination: Path,
    ledger: list[dict[str, Any]],
    *,
    source_label: str | None = None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copy hash mismatch: {source} -> {destination}")
    ledger.append(
        {
            "source": source_label or source.relative_to(REPO_ROOT).as_posix(),
            "record_path": destination.as_posix(),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    )


def build_composite(record_root: Path, panel_manifest: list[dict[str, str]]) -> None:
    if not ARIAL_BOLD_FONT.is_file():
        raise FileNotFoundError(
            f"Arial Bold is required for panel labels but was not found: {ARIAL_BOLD_FONT}"
        )
    page_width_in = 7.5
    page_height_in = 7.5
    row_top = {0: 0.35, 1: 2.55, 2: 4.55}
    gutter = 0.15
    per_panel = {
        item["panel"]: item
        for item in panel_manifest
        if item["format"] == "pdf"
    }
    placements: list[dict[str, Any]] = []
    document = fitz.open()
    page = document.new_page(width=page_width_in * 72, height=page_height_in * 72)
    page.insert_font(fontname="ArialBold", fontfile=str(ARIAL_BOLD_FONT))
    rows_by_order = (("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i"))
    for row_index, panel_row in enumerate(rows_by_order):
        widths = [float(per_panel[panel]["width_in"]) for panel in panel_row]
        total_width = sum(widths) + gutter * (len(panel_row) - 1)
        x = (page_width_in - total_width) / 2
        y = row_top[row_index]
        for column_index, panel in enumerate(panel_row):
            item = per_panel[panel]
            width = float(item["width_in"])
            height = float(item["height_in"])
            source_path = record_root / item["path"]
            source = fitz.open(source_path)
            if source.page_count != 1:
                raise ValueError(f"Expected one-page panel PDF: {source_path}")
            source_rect = source[0].rect
            if not (
                math.isclose(source_rect.width / 72, width, abs_tol=0.015)
                and math.isclose(source_rect.height / 72, height, abs_tol=0.015)
            ):
                raise ValueError(
                    f"Panel {panel} PDF size {source_rect.width / 72:.3f} x "
                    f"{source_rect.height / 72:.3f} does not match manifest {width} x {height}"
                )
            rectangle = fitz.Rect(x * 72, y * 72, (x + width) * 72, (y + height) * 72)
            page.show_pdf_page(rectangle, source, 0, keep_proportion=True)
            source.close()
            page.insert_text(
                (max(0.025, x - 0.075) * 72, (y + 0.12) * 72),
                panel,
                fontsize=10.5,
                fontname="ArialBold",
                color=(0, 0, 0),
            )
            placements.append(
                {
                    "panel": panel,
                    "role": PANEL_ROLES[panel],
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "x_in": x,
                    "y_in": y,
                    "width_in": width,
                    "height_in": height,
                    "input_path": item["path"],
                    "input_sha256": sha256(source_path),
                }
            )
            x += width + gutter

    pdf_path = record_root / "ed02_slide_inventory_version_record.pdf"
    document.save(pdf_path, garbage=4, deflate=True)
    document.close()
    with fitz.open(pdf_path) as rendered:
        if rendered.page_count != 1:
            raise RuntimeError("Composite PDF is not one page")
        svg_path = record_root / "ed02_slide_inventory_version_record.svg"
        svg_path.write_text(
            rendered[0].get_svg_image(text_as_path=False), encoding="utf-8"
        )
        pixmap = rendered[0].get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
        png_path = record_root / "ed02_slide_inventory_version_record_600dpi.png"
        pixmap.save(png_path)
    tiff_path = record_root / "ed02_slide_inventory_version_record_600dpi.tiff"
    with Image.open(png_path) as image:
        image.save(tiff_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
    outputs = {
        path.suffix.lstrip("."): {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (pdf_path, svg_path, png_path, tiff_path)
    }
    write_json(
        record_root / "ED02_COMPOSITE_MANIFEST.json",
        {
            "displayed_figure_number": 2,
            "source_slide": 3,
            "backend": "Python; vector PDF assembly with PyMuPDF from matplotlib panel PDFs",
            "page_inches": [page_width_in, page_height_in],
            "panel_order": list(PANEL_ROLES),
            "row_order": [list(row) for row in rows_by_order],
            "panel_label": {
                "font": "Arial Bold",
                "font_file": ARIAL_BOLD_FONT.name,
                "font_sha256": sha256(ARIAL_BOLD_FONT),
                "size_pt": 10.5,
            },
            "placements": placements,
            "outputs": outputs,
        },
    )


def _record_science_checks(record_root: Path, audit: Audit) -> None:
    e_rows = rows(record_root / "plot_ready/panel_e_shift_from_naive.csv")
    values = {
        item["metric"]: (
            float(item["return_minus_naive"]),
            float(item["second_day_minus_naive"]),
        )
        for item in e_rows
    }
    audit.check(
        "record_panel_e_values_exact",
        set(values) == set(EXPECTED_PANEL_E)
        and all(
            close(values[key][0], expected[0]) and close(values[key][1], expected[1])
            for key, expected in EXPECTED_PANEL_E.items()
        ),
        values,
    )
    natural = [
        item
        for item in rows(record_root / "analysis_inputs/natural/fixed_time_policy_rows.csv")
        if close(item["window_minutes"], 30, 1e-9)
        and as_bool(item["primary_policy_support"])
    ]
    counts = Counter(item["group"] for item in natural)
    audit.check(
        "record_panel_e_natural_support_counts",
        counts == {"naive": 23, "LTM": 58, "experienced": 74},
        dict(counts),
    )


def validate_staged_record(record_root: Path, final_record_name: str) -> dict[str, Any]:
    audit = Audit()
    panel_manifest = rows(record_root / "PANEL_MANIFEST.csv")
    table_manifest = rows(record_root / "TABLE_MANIFEST.csv")
    order = list(dict.fromkeys(item["panel"] for item in panel_manifest))
    roles = {
        panel: {item["role"] for item in panel_manifest if item["panel"] == panel}
        for panel in order
    }
    audit.check(
        "record_panel_role_order_parity",
        order == list(PANEL_ROLES)
        and all(roles[panel] == {PANEL_ROLES[panel]} for panel in PANEL_ROLES),
        {
            "order": order,
            "roles": {panel: sorted(values) for panel, values in roles.items()},
        },
    )
    audit.check(
        "record_panel_and_table_counts",
        len(panel_manifest) == 36 and len(table_manifest) == 15,
        {"panel_artifacts": len(panel_manifest), "plot_ready_tables": len(table_manifest)},
    )
    artifact_errors = []
    for item in (*panel_manifest, *table_manifest):
        error = _verify_manifest_file(record_root, item)
        if error:
            artifact_errors.append(error)
    audit.check("record_artifacts_match_manifests", not artifact_errors, artifact_errors)
    input_errors = []
    for key, (relative, expected_hash) in ANALYSIS_INPUTS.items():
        target = record_root / "analysis_inputs" / relative
        if not target.is_file() or sha256(target) != expected_hash:
            input_errors.append(key)
    audit.check("record_analysis_inputs_exact_13", not input_errors, input_errors)
    provenance_count = len(list((record_root / "analysis_provenance").rglob("*")))
    provenance_files = [
        path for path in (record_root / "analysis_provenance").rglob("*") if path.is_file()
    ]
    audit.check(
        "record_analysis_provenance_six_files",
        len(provenance_files) == 6,
        {"file_count": len(provenance_files), "tree_entries": provenance_count},
    )
    composite = json.loads(
        (record_root / "ED02_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit.check(
        "composite_one_page_three_by_three_order",
        composite.get("panel_order") == list(PANEL_ROLES)
        and composite.get("row_order") == [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        and len(composite.get("placements", [])) == 9,
        {"panel_order": composite.get("panel_order"), "row_order": composite.get("row_order")},
    )
    with fitz.open(record_root / "ed02_slide_inventory_version_record.pdf") as document:
        pdf_ok = document.page_count == 1 and math.isclose(document[0].rect.width, 540, abs_tol=0.1)
        label_spans = [
            span
            for block in document[0].get_text("dict").get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text") in PANEL_ROLES
            and math.isclose(float(span.get("size", 0)), 10.5, abs_tol=0.05)
            and "Arial" in str(span.get("font", ""))
        ]
    audit.check("composite_pdf_one_square_page", pdf_ok, "7.5 x 7.5 inches")
    audit.check(
        "composite_panel_labels_arial_bold_10p5",
        {span["text"] for span in label_spans} == set(PANEL_ROLES),
        [
            {"text": span["text"], "font": span.get("font"), "size": span.get("size")}
            for span in label_spans
        ],
    )
    with Image.open(record_root / "ed02_slide_inventory_version_record_600dpi.png") as image:
        png_size = image.size
    audit.check("composite_png_is_600dpi_canvas", png_size == (4500, 4500), png_size)
    _record_science_checks(record_root, audit)
    forbidden = [
        path.relative_to(record_root).as_posix()
        for path in record_root.rglob("*")
        if path.is_file() and "ed02j" in path.name.lower()
    ]
    audit.check("record_contains_no_panel_j", not forbidden, forbidden)
    render_result = json.loads((record_root / "RENDER_RESULT.json").read_text(encoding="utf-8"))
    audit.check(
        "render_result_uses_record_local_portable_authority",
        render_result.get("output_root") == "record-local copied render payload",
        render_result.get("output_root"),
    )
    version_manifest = json.loads(
        (record_root / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit.check(
        "version_manifest_uses_final_stable_record_name",
        version_manifest.get("record") == final_record_name
        and ".staging-" not in json.dumps(version_manifest),
        version_manifest.get("record"),
    )
    return audit.result()


def _manifest_text(record_name: str) -> str:
    return f"""# Extended Data Figure 2 | slide-defined nine-panel version record

Promoted on 2026-08-08 as an immutable scientific-figure record for the supplied
PowerPoint slide 3 inventory. The accepted inventory is exactly **a--i** in
three rows: `a,b,c`; `d,e,f`; `g,h,i`. A broader ten-panel audit page is not
the slide authority and panel j is intentionally absent.

The panels use the first 30 cumulative minutes of all-mode TimeInMaze while
policy events are Explore-mode junction decisions. Panel e retains the
PowerPoint role, "Shift from naive," and is recomputed from the current natural
primary-policy support (Naive n=23, Return n=58, Second day n=74).

This analysis is behavior-only. Manual neuronal masks, CIMaps, footprint files,
and sorted-neuron exclusions are not in its dependency closure.

`RECORD_VALIDATION.json` and `PROMOTION_COMPLETE.txt` were written before
`RECORD_HASHES.csv`; the latter covers every file in `{record_name}` except
itself. Any future change requires a new version-record directory.
"""


def _legend_text() -> str:
    return """# Extended Data Fig. 2 | Experience-dependent route policies and biased-walk reconstruction

**a,** Four branch-policy definitions at non-root binary-tree junctions.
**b,** Naive versus Return policy probabilities.
**c,** Return versus Second-day policy probabilities.
**d,** One-policy-at-a-time substitution contribution test.
**e,** Return and Second-day shifts from the Naive policy means, using natural
primary-policy support (Naive n=23 sessions, Return n=58, Second day n=74).
**f,** Four-policy walk calibration.
**g,** Depth-dependent stemward-exit policy profile and contrasts.
**h,** Four-policy versus depth-varying reconstruction of the Naive-to-Return gap.
**i,** Depth-varying walk calibration.

The clock is the first 30 cumulative minutes of all-mode TimeInMaze from
session start; policy events are Explore-mode junction decisions. Panels b and
d use strict equal-junction support where declared; c, e--g and i use the
declared natural/calibration supports. Confidence intervals, cluster units,
multiple-comparison corrections and calibration gates are specified in the
plot-ready tables and frozen analysis reports. Reconstruction is
contemporaneous, not prospective or causal, and is not variance explained.
"""


def _version_manifest(
    record_root: Path, final_record_name: str, ledger: list[dict[str, Any]]
) -> dict[str, Any]:
    panel_e = rows(record_root / "plot_ready/panel_e_shift_from_naive.csv")
    return {
        "figure": "Extended Data Figure 2",
        "status": "accepted_immutable_version_record",
        "accepted": "2026-08-08",
        "accepted_round": "slide_inventory_correction_round2",
        "record": final_record_name,
        "supersedes_unpointed_candidates": [
            "extended_data_fig02_2026-08-08_round2_slide_inventory",
            "extended_data_fig02_2026-08-08_round2_slide_inventory_portable",
        ],
        "source_slide": 3,
        "panel_inventory_authority": "supplied PowerPoint slide 3",
        "panel_order": list(PANEL_ROLES),
        "panel_roles": PANEL_ROLES,
        "row_order": [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
        "core_conclusion": (
            "During the first 30 cumulative minutes in the maze, Explore-mode "
            "junction policies differ with experience; stationary four-policy and "
            "depth-varying biased walks provide complementary contemporaneous "
            "reconstructions of the Naive-to-Return efficiency gap."
        ),
        "archetype": "schematic-led quantitative 3 x 3 grid",
        "backend": "Python only; matplotlib panels and Python/PyMuPDF composite",
        "timebase": (
            "first 30 cumulative minutes of all-mode TimeInMaze from session start; "
            "Explore-mode junction decisions only"
        ),
        "representation": "non-root T-junction branch decisions on the binary-tree maze",
        "reward_context": "behavioral exploration analysis; no reward-endpoint neural map",
        "manual_mask_applicability": "not applicable; behavior-only analysis",
        "panel_e": {
            "role": PANEL_ROLES["e"],
            "support": "natural primary-policy support",
            "session_counts": {"Naive": 23, "Return": 58, "Second day": 74},
            "values": panel_e,
        },
        "claim_limits": [
            "contemporaneous reconstruction, not prospective prediction",
            "not variance explained, causal mediation, biological sufficiency, or causal share",
            "fixed donor means and Monte Carlo walk noise are not propagated through intervals",
            "depth-model calibration passes for Naive and Return but not Second day",
        ],
        "export_contract": {
            "individual_panels": 9,
            "formats_per_panel": list(FORMATS),
            "panel_artifacts": 36,
            "plot_ready_tables": 15,
            "renderer_input_tables": 13,
            "raster_dpi": 600,
            "editable_panel_svg_text": True,
            "composite_page_inches": [7.5, 7.5],
        },
        "promotion_payload_file": "AUDIT_PROMOTION.json",
    }


def promote(source_root: Path, source_record: Path, record_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_record = source_record.resolve()
    record_root = record_root.resolve()
    if record_root.exists():
        raise FileExistsError(f"Refusing to overwrite immutable record: {record_root}")
    source_validation = validate_source(source_root, source_record)
    if source_validation["status"] != "pass":
        raise RuntimeError(f"Source validation failed: {source_validation}")
    record_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{record_root.name}.staging-", dir=record_root.parent)
    )
    ledger: list[dict[str, Any]] = []
    try:
        panel_manifest_source = rows(source_root / "PANEL_MANIFEST.csv")
        table_manifest_source = rows(source_root / "TABLE_MANIFEST.csv")
        for item in panel_manifest_source:
            copy_file(
                source_root / item["path"],
                staging / item["path"],
                ledger,
                source_label=f"renderer_output/{item['path']}",
            )
        for item in table_manifest_source:
            copy_file(
                source_root / item["path"],
                staging / item["path"],
                ledger,
                source_label=f"renderer_output/{item['path']}",
            )
        for name in ("PANEL_MANIFEST.csv", "TABLE_MANIFEST.csv", "SOURCE_MANIFEST.json"):
            copy_file(
                source_root / name,
                staging / name,
                ledger,
                source_label=f"renderer_output/{name}",
            )
        render_result = json.loads(
            (source_root / "RENDER_RESULT.json").read_text(encoding="utf-8")
        )
        render_result["source_output_sha256"] = sha256(source_root / "RENDER_RESULT.json")
        render_result["output_root"] = "record-local copied render payload"
        render_result["metadata_normalization"] = (
            "Transient source output_root removed during immutable promotion; panel and "
            "table paths are record-local in PANEL_MANIFEST.csv and TABLE_MANIFEST.csv."
        )
        write_json(staging / "RENDER_RESULT.json", render_result)
        ledger.append(
            {
                "source": "normalized from final source RENDER_RESULT.json",
                "record_path": "RENDER_RESULT.json",
                "sha256": sha256(staging / "RENDER_RESULT.json"),
                "bytes": (staging / "RENDER_RESULT.json").stat().st_size,
            }
        )

        input_manifest_rows = []
        for key, (relative, expected_hash) in ANALYSIS_INPUTS.items():
            source = source_record / "analysis_inputs" / relative
            destination = staging / "analysis_inputs" / relative
            copy_file(source, destination, ledger)
            input_manifest_rows.append(
                {
                    "source_key": key,
                    "record_path": f"analysis_inputs/{relative}",
                    "source_record_path": (
                        f"docs/figure_version_records/{source_record.name}/analysis_inputs/{relative}"
                    ),
                    "sha256": expected_hash,
                    "bytes": source.stat().st_size,
                }
            )
        write_csv(
            staging / "INPUT_MANIFEST.csv",
            ("source_key", "record_path", "source_record_path", "sha256", "bytes"),
            input_manifest_rows,
        )
        copy_file(
            source_record / "INPUT_MANIFEST.csv",
            staging / "UPSTREAM_INPUT_MANIFEST.csv",
            ledger,
        )
        for relative in PROVENANCE_FILES:
            copy_file(
                source_record / "analysis_provenance" / relative,
                staging / "analysis_provenance" / relative,
                ledger,
            )
        for source in SOURCE_CODE:
            copy_file(source, staging / "source_code" / source.name, ledger)

        build_composite(staging, rows(staging / "PANEL_MANIFEST.csv"))
        for path in (
            staging / "ed02_slide_inventory_version_record.pdf",
            staging / "ed02_slide_inventory_version_record.svg",
            staging / "ed02_slide_inventory_version_record_600dpi.png",
            staging / "ed02_slide_inventory_version_record_600dpi.tiff",
            staging / "ED02_COMPOSITE_MANIFEST.json",
        ):
            ledger.append(
                {
                    "source": "generated from record-local nine panel PDFs",
                    "record_path": path.relative_to(staging).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

        # Normalize copied paths to their final record-relative destinations.
        for item in ledger:
            path = Path(str(item["record_path"]))
            if path.is_absolute():
                item["record_path"] = path.relative_to(staging).as_posix()
            else:
                item["record_path"] = path.as_posix()
        write_json(staging / "AUDIT_PROMOTION.json", ledger)
        (staging / "MANIFEST.md").write_text(
            _manifest_text(record_root.name), encoding="utf-8"
        )
        (staging / "FIGURE_LEGEND.md").write_text(_legend_text(), encoding="utf-8")
        (staging / "REPORT.md").write_text(
            "# ED2 slide-inventory correction report\n\n"
            "The record restores the supplied slide's nine scientific roles a--i. "
            "Panel e is the current fixed-30-minute Shift-from-naive analysis; the "
            "ten-panel audit-page replacement and panel j are excluded. All 36 panel "
            "exports, 15 plot-ready tables, 13 renderer inputs, six provenance files, "
            "source code, and a one-page 3 x 3 composite are frozen here.\n",
            encoding="utf-8",
        )
        write_json(
            staging / "ED02_VERSION_RECORD_MANIFEST.json",
            _version_manifest(staging, record_root.name, ledger),
        )

        staged_validation = validate_staged_record(staging, record_root.name)
        staged_validation["source_validation"] = source_validation
        staged_validation["hash_policy"] = (
            "RECORD_VALIDATION.json and PROMOTION_COMPLETE.txt are written before "
            "RECORD_HASHES.csv; the ledger then covers every record file except itself."
        )
        write_json(staging / "RECORD_VALIDATION.json", staged_validation)
        if staged_validation["status"] != "pass":
            raise RuntimeError(f"Staged record validation failed: {staged_validation}")
        (staging / "PROMOTION_COMPLETE.txt").write_text(
            "PROMOTED "
            + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            + " after nine-panel role/order and fixed30 source audit.\n",
            encoding="utf-8",
        )

        # This must remain the final write inside the record.
        hash_rows = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "RECORD_HASHES.csv":
                hash_rows.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
        write_csv(staging / "RECORD_HASHES.csv", ("path", "sha256", "bytes"), hash_rows)
        ledger_paths = {item["path"] for item in rows(staging / "RECORD_HASHES.csv")}
        actual_paths = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name != "RECORD_HASHES.csv"
        }
        if ledger_paths != actual_paths:
            raise RuntimeError("Hash ledger does not cover every record file")
        for item in rows(staging / "RECORD_HASHES.csv"):
            target = staging / item["path"]
            if sha256(target) != item["sha256"] or target.stat().st_size != int(item["bytes"]):
                raise RuntimeError(f"Hash closure failed for {item['path']}")
        staging.replace(record_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "status": "pass",
        "record": record_root.relative_to(REPO_ROOT).as_posix(),
        "panel_order": list(PANEL_ROLES),
        "panel_artifacts": 36,
        "plot_ready_tables": 15,
        "analysis_inputs": 13,
        "hash_rows": len(rows(record_root / "RECORD_HASHES.csv")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-record", type=Path, default=CURRENT_SOURCE_RECORD)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the final source panels and frozen inputs without writing a record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        result = validate_source(args.source_root.resolve(), args.source_record.resolve())
    else:
        result = promote(args.source_root, args.source_record, args.record_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
