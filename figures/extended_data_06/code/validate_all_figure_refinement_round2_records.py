"""Independent acceptance for the 2026-08-11 all-figure refinement round."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
OUTPUT = RECORDS / "ALL_FIGURE_REFINEMENT_ROUND2_VALIDATION.json"
MAIN_POINTER = RECORDS / "CURRENT_MAIN_FIGURES.md"
ED_POINTER = RECORDS / "CURRENT_EXTENDED_DATA_FIGURES.md"

EXPECTED = {
    "main1": "main_figure1_2026-08-13_round12_no_port_mouse1636",
    "main2": "main_figure2_2026-08-13_round5_pair_mean_day1day2",
    "main3": "main_figure3_2026-08-14_round8_bh4_correlations",
    "main4": "main_figure4_2026-08-15_round7_neuron_id_shuffle",
    "main5": "main_figure5_2026-08-15_round8_neuron_id_shuffle",
    "ed1": "extended_data_fig01_2026-08-15_round18_group80_matched153",
    "ed2": "extended_data_fig02_2026-08-16_round26_depth_common_i",
    "ed3": "extended_data_fig03_2026-08-13_round3_latent_day_labels",
    "ed4": "extended_data_fig04_2026-08-12_round4_cycle_labels",
    "ed5": "extended_data_fig05_2026-08-12_round14_full_place_maps",
    "ed6": "extended_data_fig06_2026-08-16_round6_upright_i",
    "ed7": "extended_data_fig07_2026-08-16_round16_upright_i_axis",
    "ed8": "extended_data_fig08_2026-08-11_round13_population_vector_label",
    "ed9": "extended_data_fig09_2026-08-14_round32_bh4_correlations",
    "ed10": "extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order",
}

ROOT_STEMS = {
    **{f"main{i}": f"figure{i}_version_record" for i in range(1, 6)},
    **{f"ed{i}": f"ed{i:02d}_version_record" for i in range(1, 11)},
}

DISPLAY_ASSERTIONS = {
    "main1": {"present": ("Time", "First visit route score", "Next cycle: i \u2192 i + 90", "Fraction of sessions"), "absent": ("First-visit route score", "Time (months)", "Next session", "No. of sessions")},
    "main3": {"present": ("No. of cycles",), "absent": ("# of experiences", "No. of experiences")},
    "main4": {"present": ("Day i place maps", "Day i + 90 activity"), "absent": ()},
    "main5": {"present": ("Normalized rate",), "absent": ("Norm. activity",)},
    "ed1": {"present": ("Time in maze", "End node visits", "End node visit", "Coverage round"), "absent": ("Maze time", "Endnode", "Cycle efficiency", "DejaVu Sans")},
    "ed3": {"present": ("Latent day2", "Reward-path minus other dwell", "Mean reward-path junction dwell", "Reward-path approaches after reward"), "absent": ("Latent Day 2", "Direct-minus-other dwell", "Mean direct-chain junction dwell", "Post-port direct approach")},
    "ed4": {"present": ("No. of cycles",), "absent": ("# of experiences", "No. of experiences")},
    "ed5": {"present": (), "absent": ("Norm. activity",)},
    "ed6": {"present": ("Population vector correlation", r"Day $\mathrm{i}$", r"Day $\mathrm{i}+1$"), "absent": ("PV correlation", "Day $i$", "Day $i+1$", "Day 1", "Day 2")},
    "ed7": {"present": ("Cycle 0 (naive)", "Cycle 1 (0-3 mo)", r"Day $\mathrm{i}$ early vs Day $\mathrm{i}+1$ correlation"), "absent": ("Exp0 (naive)", "Exp1 (0-3 mo)", "Experience\n", "Transition\n", "Day $i$ early vs Day $i+1$ correlation", "Day 1 early vs Day 2 correlation")},
    "ed8": {"present": ("Population vector correlation",), "absent": ("PV correlation",)},
    "ed9": {"present": ("No. of cycles", "Vertical 8-bin", "−0.1", "0.1"), "absent": ("# of experiences", "No. of experiences", "−10.00%", "+10.00%")},
    "ed10": {"present": ("Depth-weighted",), "absent": ()},
}

GLOBAL_FORBIDDEN = (
    "maze-tile",
    "long-bout",
    "long-gap",
    "first-long bout",
    "# of experiences",
    "Norm. activity",
    "First-visit route score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_relative_files(root: Path) -> list[tuple[str, Path]]:
    """Enumerate every record file, including Windows paths beyond MAX_PATH."""
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


def verify_hashes(record: Path) -> tuple[bool, list[str]]:
    inventory = record / "RECORD_HASHES.csv"
    if not inventory.is_file():
        return False, ["missing-ledger"]
    with inventory.open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        relative: path
        for relative, path in _iter_relative_files(record)
        if relative != inventory.name
    }
    errors = [f"unhashed:{key}" for key in sorted(set(actual) - set(expected))]
    errors += [f"missing:{key}" for key in sorted(set(expected) - set(actual))]
    for key in sorted(set(actual) & set(expected)):
        path = actual[key]
        row = expected[key]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{key}")
    return not errors, errors


def combined_visible_text(svg: Path, pdf: Path) -> str:
    text = svg.read_text(encoding="utf-8")
    with fitz.open(pdf) as document:
        text += "\n" + "\n".join(page.get_text() for page in document)
    return text


def root_exports(record: Path, key: str) -> dict[str, Path]:
    stem = ROOT_STEMS[key]
    return {
        "svg": record / f"{stem}.svg",
        "pdf": record / f"{stem}.pdf",
        "png300": record / f"{stem}_300dpi.png",
        "png600": record / f"{stem}_600dpi.png",
        "tiff600": record / f"{stem}_600dpi.tiff",
    }


def export_checks(paths: dict[str, Path]) -> tuple[dict[str, bool], dict[str, Any]]:
    checks = {f"{name}_exists": path.is_file() for name, path in paths.items()}
    details: dict[str, Any] = {}
    if not all(checks.values()):
        return checks, details
    with fitz.open(paths["pdf"]) as document:
        checks["pdf_one_page"] = document.page_count == 1
        page = document[0].rect
        expected600 = (round(page.width / 72 * 600), round(page.height / 72 * 600))
        page_points = (page.width, page.height)
        details["pdf_page_pt"] = [page.width, page.height]
    for role, dpi in (("png300", 300), ("png600", 600), ("tiff600", 600)):
        if role not in paths:
            continue
        with Image.open(paths[role]) as image:
            details[f"{role}_size"] = list(image.size)
            details[f"{role}_dpi"] = [float(value) for value in image.info.get("dpi", (0, 0))]
            if dpi == 600:
                checks[f"{role}_pixel_size_matches_pdf"] = all(
                    abs(actual - expected) <= max(1, round(expected * 0.002))
                    for actual, expected in zip(image.size, expected600)
                )
                saved_dpi = image.info.get("dpi", (0, 0))
                inferred_dpi = tuple(
                    actual * 72 / points
                    for actual, points in zip(image.size, page_points)
                )
                effective_dpi = (
                    saved_dpi if saved_dpi and min(saved_dpi) > 0 else inferred_dpi
                )
                details[f"{role}_effective_dpi"] = [
                    float(value) for value in effective_dpi
                ]
                checks[f"{role}_dpi_at_least_590"] = min(effective_dpi) >= 590
    return checks, details


def manifest_hash_checks(record: Path, paths: dict[str, Path]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for manifest in record.glob("*VERSION_RECORD_MANIFEST.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        checks[f"{manifest.stem}_record_identity"] = payload.get("record") == record.name
        for field in ("root_export_sha256", "root_output_sha256"):
            values = payload.get(field)
            if not isinstance(values, dict):
                continue
            role_map = {
                "svg": "svg",
                "pdf": "pdf",
                "png_300dpi": "png300",
                "png_600dpi": "png600",
                "tiff_600dpi": "tiff600",
                "tiff_600dpi_lzw": "tiff600",
            }
            for role, expected in values.items():
                if role in role_map:
                    checks[f"{manifest.stem}_{field}_{role}"] = sha256(paths[role_map[role]]) == expected
    return checks


def validate() -> dict[str, Any]:
    main_pointer = MAIN_POINTER.read_text(encoding="utf-8")
    ed_pointer = ED_POINTER.read_text(encoding="utf-8")
    results: dict[str, dict[str, bool]] = {}
    details: dict[str, Any] = {}
    violations: list[str] = []
    for key, name in EXPECTED.items():
        record = RECORDS / name
        pointer = main_pointer if key.startswith("main") else ed_pointer
        checks: dict[str, bool] = {
            "record_exists": record.is_dir(),
            "pointer_selects_record": f"{name}/" in pointer,
        }
        if not record.is_dir():
            results[key] = checks
            continue
        hash_ok, hash_errors = verify_hashes(record)
        checks["complete_hash_closure"] = hash_ok
        details[f"{key}_hash_errors"] = hash_errors
        paths = root_exports(record, key)
        if key == "ed3":
            paths.pop("png300")
        extra, export_detail = export_checks(paths)
        checks.update(extra)
        details[f"{key}_exports"] = export_detail
        if paths["svg"].is_file() and paths["pdf"].is_file():
            text = combined_visible_text(paths["svg"], paths["pdf"])
            for token in GLOBAL_FORBIDDEN:
                if token.lower() in text.lower():
                    violations.append(f"{key}:{token}")
            assertion = DISPLAY_ASSERTIONS.get(key, {"present": (), "absent": ()})
            for token in assertion["present"]:
                checks[f"visible_present_{re.sub(r'[^a-z0-9]+', '_', token.lower()).strip('_')}"] = token in text
            for token in assertion["absent"]:
                checks[f"visible_absent_{re.sub(r'[^a-z0-9]+', '_', token.lower()).strip('_')}"] = token not in text
            checks["obsolete_16897_absent"] = re.search(r"(?<!\d)16897(?!\d)", text) is None
        human_metadata = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                record / "CAPTION_NOTES.md",
                record / "MANIFEST.md",
                record / "REBUILD_README.md",
            )
            if path.is_file()
        )
        metadata_violations = [
            token for token in GLOBAL_FORBIDDEN if token.lower() in human_metadata.lower()
        ]
        checks["current_human_metadata_forbidden_terms_absent"] = not metadata_violations
        violations.extend(f"{key}:metadata:{token}" for token in metadata_violations)
        checks.update(manifest_hash_checks(record, paths))
        if key == "ed5":
            renderer = record / "source_code/render_ed05_example_place_maps_assembly.py"
            checks["normalized_rate_locked_in_native_renderer"] = (
                renderer.is_file() and "Normalized rate" in renderer.read_text(encoding="utf-8")
            )
        if key == "ed9":
            stats_path = (
                record
                / "source_data/panel_f_inference_round24/be_roles/"
                "B2_EXCLUSION_CLEAN32_CR1_BH2_STATISTICS.csv"
            )
            if stats_path.is_file():
                with stats_path.open(newline="", encoding="utf-8") as stream:
                    stats_rows = {row["contrast"]: row for row in csv.DictReader(stream)}
                expected_q = {
                    "geometric8_minus_horizontal8": "2.8670561454901405e-05",
                    "geometric8_minus_vertical8": "1.6728044637612076e-06",
                }
                checks["panel_f_exact_two_contrasts"] = set(stats_rows) == set(expected_q)
                checks["panel_f_clean32_14_t13"] = all(
                    row.get("n_pairs") == "32"
                    and row.get("n_mice") == "14"
                    and row.get("cluster_df") == "13"
                    for row in stats_rows.values()
                )
                checks["panel_f_exact_bh2_values"] = all(
                    stats_rows.get(contrast, {}).get("q_bh_two_raw_planned_contrasts") == q_value
                    for contrast, q_value in expected_q.items()
                )
            else:
                checks["panel_f_exact_two_contrasts"] = False
                checks["panel_f_clean32_14_t13"] = False
                checks["panel_f_exact_bh2_values"] = False
        if key not in {"main1", "ed3"}:
            refinement = record / "ROUND2_DISPLAY_REFINEMENT_MANIFEST.json"
            checks["refinement_manifest_pass"] = refinement.is_file() and json.loads(
                refinement.read_text(encoding="utf-8")
            ).get("status") == "PASS"
        elif key == "main1":
            refinement = record / "AF23_DIRECT_ENTRY_RECONSTRUCTION_MANIFEST.json"
            checks["refinement_manifest_pass"] = refinement.is_file() and json.loads(
                refinement.read_text(encoding="utf-8")
            ).get("status") == "PASS"
        results[key] = checks
    caption_root = REPO / "docs/manuscript_figure_captions_2026-08-10"
    caption_text = "\n".join(
        (caption_root / name).read_text(encoding="utf-8")
        for name in (
            "figure_captions_main1-5_extended1-10.tex",
            "TERMINOLOGY_LEDGER.md",
            "CAPTION_EVIDENCE_AUDIT.md",
        )
    )
    for token in GLOBAL_FORBIDDEN:
        if token.lower() in caption_text.lower():
            violations.append(f"captions:{token}")
    global_checks = {
        "all_expected_records_named_in_caption_evidence": all(
            name in caption_text for name in EXPECTED.values()
        ),
        "global_forbidden_visible_terms_absent": not violations,
    }
    failures = [
        f"{key}:{check}"
        for key, checks in results.items()
        for check, passed in checks.items()
        if not passed
    ]
    failures += [name for name, passed in global_checks.items() if not passed]
    return {
        "schema": "all-figure-refinement-round2-validation/v1",
        "status": "PASS" if not failures else "FAIL",
        "records": EXPECTED,
        "results": results,
        "global_checks": global_checks,
        "violations": violations,
        "failures": failures,
        "details": details,
    }


def main() -> int:
    payload = validate()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": payload["failures"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
