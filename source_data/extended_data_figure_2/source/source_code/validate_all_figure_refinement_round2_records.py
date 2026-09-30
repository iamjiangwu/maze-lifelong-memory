"""Independent acceptance for the 2026-08-11 all-figure refinement round."""

from __future__ import annotations

import csv
import hashlib
import json
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
    "main1": "main_figure1_2026-08-11_round7_complete_raster_export_closure",
    "main2": "main_figure2_2026-08-11_round4_schematic_refinement",
    "main3": "main_figure3_2026-08-12_round5_embedded_raster_resolution",
    "main4": "main_figure4_2026-08-11_round5_bottom_labels",
    "main5": "main_figure5_2026-08-12_round6_historical_hierarchical_order",
    "ed1": "extended_data_fig01_2026-08-12_round13_alignment_clearance",
    "ed2": "extended_data_fig02_2026-08-12_round23_preceding_turn",
    "ed3": "extended_data_fig03_2026-08-11_round1_direct_entry_reconstruction",
    "ed4": "extended_data_fig04_2026-08-12_round3_embedded_raster_resolution",
    "ed5": "extended_data_fig05_2026-08-12_round14_full_place_maps",
    "ed6": "extended_data_fig06_2026-08-11_round4_arrow_population_vector",
    "ed7": "extended_data_fig07_2026-08-11_round13_compact_legends",
    "ed8": "extended_data_fig08_2026-08-11_round13_population_vector_label",
    "ed9": "extended_data_fig09_2026-08-12_round26_margin_star_density_resolution",
    "ed10": "extended_data_fig10_2026-08-11_round10_upright_policy_labels",
}

ROOT_STEMS = {
    **{f"main{i}": f"figure{i}_version_record" for i in range(1, 6)},
    **{f"ed{i}": f"ed{i:02d}_version_record" for i in range(1, 11)},
}

DISPLAY_ASSERTIONS = {
    "main1": {"present": ("Time", "First visit route score"), "absent": ("First-visit route score", "Time (months)")},
    "main3": {"present": ("No. of experiences",), "absent": ("# of experiences",)},
    "main4": {"present": ("Day i place maps", "Day i + 90 activity"), "absent": ()},
    "main5": {"present": ("Normalized rate",), "absent": ("Norm. activity",)},
    "ed1": {"present": ("Time in maze", "End node visits", "End node visit"), "absent": ("Maze time", "Endnode")},
    "ed4": {"present": ("No. of experiences",), "absent": ("# of experiences",)},
    "ed5": {"present": (), "absent": ("Norm. activity",)},
    "ed6": {"present": ("Population vector correlation",), "absent": ("PV correlation",)},
    "ed7": {"present": ("Exp0 (naive)", "Exp1 (0-3 mo)"), "absent": ("Experience\n", "Transition\n")},
    "ed8": {"present": ("Population vector correlation",), "absent": ("PV correlation",)},
    "ed9": {"present": ("No. of experiences", "Vertical 8-bin"), "absent": ("# of experiences",)},
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


def verify_hashes(record: Path) -> tuple[bool, list[str]]:
    inventory = record / "RECORD_HASHES.csv"
    if not inventory.is_file():
        return False, ["missing-ledger"]
    with inventory.open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != inventory.name
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
        details["pdf_page_pt"] = [page.width, page.height]
    for role, dpi in (("png300", 300), ("png600", 600), ("tiff600", 600)):
        if role not in paths:
            continue
        with Image.open(paths[role]) as image:
            details[f"{role}_size"] = list(image.size)
            details[f"{role}_dpi"] = [float(value) for value in image.info.get("dpi", (0, 0))]
            if dpi == 600:
                checks[f"{role}_pixel_size_matches_pdf"] = all(
                    abs(actual - expected) <= 1
                    for actual, expected in zip(image.size, expected600)
                )
                checks[f"{role}_dpi_at_least_590"] = min(image.info.get("dpi", (0, 0))) >= 590
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
