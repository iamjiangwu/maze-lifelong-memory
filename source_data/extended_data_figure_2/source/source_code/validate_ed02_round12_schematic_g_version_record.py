#!/usr/bin/env python3
"""Independent read-only validator for the corrected ED2 Round-12 record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RECORD = "extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor"
DEFAULT_RECORD_ROOT = REPO_ROOT / "docs" / "figure_version_records" / EXPECTED_RECORD
STEM = "ed02_round12_schematic_g_fix_version_record"
EXPECTED_TABLES = {
    "panel_a_schematic_rows.csv", "panel_b_contrasts.csv", "panel_b_session_points.csv",
    "panel_c_contrasts.csv", "panel_c_session_points.csv", "panel_d_policy_contributions.csv",
    "panel_e_distance_from_return.csv", "panel_e_group_means.csv", "panel_f_calibration_summary.csv",
    "panel_f_session_points.csv", "panel_g_depth_contrasts.csv", "panel_g_group_profiles.csv",
    "panel_g_supported_session_rows.csv", "panel_h_four_vs_depth_gap.csv",
    "panel_i_calibration_summary.csv", "panel_i_session_points.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, passed: bool, detail: object) -> None:
        self.checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    def result(self) -> dict[str, object]:
        failures = [item for item in self.checks if item["status"] == "fail"]
        return {"status": "pass" if not failures else "fail", "check_count": len(self.checks), "failure_count": len(failures), "checks": self.checks}


def validate(root: Path) -> dict[str, object]:
    audit = Audit()
    audit.check("record_exists", root.is_dir(), str(root))
    if not root.is_dir():
        return audit.result()

    ledger = list(csv.DictReader((root / "RECORD_HASHES.csv").open(encoding="utf-8-sig", newline="")))
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "RECORD_HASHES.csv")
    ledger_paths = [row["path"] for row in ledger]
    audit.check("hash_closure", sorted(ledger_paths) == actual and len(ledger_paths) == len(set(ledger_paths)), {"ledger": len(ledger_paths), "actual": len(actual)})
    for row in ledger:
        path = root / row["path"]
        audit.check(f"hash::{row['path']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])

    version = json.loads((root / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("record_identity", root.name == EXPECTED_RECORD and version.get("record") == EXPECTED_RECORD and version.get("status") == "accepted_version_record", version.get("record"))
    audit.check("panel_inventory", version.get("panel_inventory") == "a-i exactly; no panel j" and version.get("panel_count") == 9, version.get("panel_inventory"))
    audit.check("behavior_only_mask_na", "not applicable" in version.get("manual_mask_applicability", "").lower(), version.get("manual_mask_applicability"))
    audit.check("scientific_values_unchanged", version.get("scientific_values_changed") is False, version.get("scientific_values_changed"))
    audit.check("fixed30_explore", version.get("window_minutes") == 30 and "Explore-mode" in version.get("policy_events", ""), {"window": version.get("window_minutes"), "events": version.get("policy_events")})

    panels = sorted((root / "panels").glob("*"))
    audit.check("panel_exports_36", len(panels) == 36 and {path.suffix.lower() for path in panels} == {".svg", ".pdf", ".png", ".tiff"}, len(panels))
    panel_rows = list(csv.DictReader((root / "PANEL_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("panel_manifest_36", len(panel_rows) == 36 and {row["panel"] for row in panel_rows} == set("abcdefghi"), len(panel_rows))
    for row in panel_rows:
        path = root / row["path"]
        audit.check(f"panel_manifest::{row['path']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])
    tables = {path.name for path in (root / "plot_ready").glob("*.csv")}
    audit.check("tables_exact_16", tables == EXPECTED_TABLES, sorted(tables))
    table_rows = list(csv.DictReader((root / "TABLE_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("table_manifest_16", len(table_rows) == 16 and {Path(row["path"]).name for row in table_rows} == EXPECTED_TABLES, len(table_rows))
    for row in table_rows:
        path = root / row["path"]
        audit.check(f"table_manifest::{row['path']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])

    inputs = list(csv.DictReader((root / "INPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("analysis_inputs_13", len(inputs) == 13 and len({row["source_key"] for row in inputs}) == 13, len(inputs))
    for row in inputs:
        path = root / row["record_path"]
        audit.check(f"analysis_input::{row['source_key']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])
    provenance = [path for path in (root / "analysis_provenance").rglob("*") if path.is_file()]
    audit.check("analysis_provenance_6", len(provenance) == 6, len(provenance))

    source = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("source_manifest_identity", source.get("status") == "accepted_version_record" and source.get("record") == EXPECTED_RECORD, {"status": source.get("status"), "record": source.get("record")})
    panel_a = source.get("panel_a_display_contract_round12", {})
    panel_g = source.get("panel_g_display_contract_round12", {})
    audit.check("panel_a_scaled_artists", panel_a.get("grid_scale_vs_round10") == 0.8 and panel_a.get("point_artist_scale_vs_round11") == 0.8 and panel_a.get("style", {}).get("arrow_mutation_scale") == 6.0 and panel_a.get("style", {}).get("arrow_linewidth") == 1.0, panel_a)
    audit.check("panel_a_annotation_clear_contract", panel_a.get("preceding_turn", {}).get("text") == "preceding turn" and panel_a.get("preceding_turn", {}).get("must_not_overlap_grid") is True, panel_a.get("preceding_turn"))
    audit.check("panel_g_spacing_clearance_contract", panel_g.get("group_position_scale_vs_round11") == 0.72 and panel_g.get("group_positions") == [0.28, 1.0, 1.72] and panel_g.get("source_values_changed") is False, panel_g)
    authority_text = "\n".join((root / name).read_text(encoding="utf-8") for name in ("SOURCE_MANIFEST.json", "RENDER_RESULT.json", "ED02_VERSION_RECORD_MANIFEST.json"))
    audit.check("portable_authority", "\\tmp\\" not in authority_text.lower() and "/tmp/" not in authority_text.lower() and "D:\\" not in authority_text, "record-local")

    composite_manifest = json.loads((root / "ED02_ROUND12_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("composite_manifest", composite_manifest.get("status") == "pass" and composite_manifest.get("review_round") == 12 and composite_manifest.get("panel_order") == list("abcdefghi"), composite_manifest.get("panel_order"))
    outputs = {
        "pdf": f"{STEM}.pdf", "svg": f"{STEM}.svg", "png": f"{STEM}_600dpi.png", "tiff": f"{STEM}_600dpi.tiff",
    }
    for key, name in outputs.items():
        row = composite_manifest["outputs"][key]
        path = root / name
        audit.check(f"composite_hash::{key}", row["path"] == name and path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row)
    with fitz.open(root / outputs["pdf"]) as document:
        page = document[0]
        audit.check("primary_true_vector_pdf", document.page_count == 1 and page.rect.width == 540 and page.rect.height == 540 and len(page.get_images(full=True)) == 0 and len(page.get_drawings()) > 900 and len(page.get_text()) > 500, {"pages": document.page_count, "images": len(page.get_images(full=True)), "drawings": len(page.get_drawings()), "characters": len(page.get_text())})
    svg_text = (root / outputs["svg"]).read_text(encoding="utf-8")
    audit.check("primary_editable_svg", svg_text.count("<text") > 100 and "<image" not in svg_text and "href=\"http" not in svg_text, {"text": svg_text.count("<text"), "image": svg_text.count("<image")})
    for key in ("png", "tiff"):
        with Image.open(root / outputs[key]) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"primary_{key}_600dpi", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})
    for path in sorted((root / "panels").glob("*.png")) + sorted((root / "panels").glob("*.tiff")):
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"panel_600dpi::{path.name}", min(dpi) >= 599.0, dpi)
    audit.check("editable_panel_svgs", all("<text" in path.read_text(encoding="utf-8") and "<image" not in path.read_text(encoding="utf-8") for path in (root / "panels").glob("*.svg")), "nine editable panels")
    audit.check("outlined_ppt_svgs_10", len(list((root / "ppt_vector").glob("*.svg"))) == 10 and all("<text" not in path.read_text(encoding="utf-8") for path in (root / "ppt_vector").glob("*.svg")), "ten outlined vectors")
    for pptx in ("sup-figs-v2_ed02-slide3_round12_review.pptx", "ed02-slide3_round12_review.pptx"):
        with zipfile.ZipFile(root / "review_evidence" / pptx) as archive:
            vectors = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
        audit.check(f"review_vector_inventory::{pptx}", len(vectors) == 10, len(vectors))
    saved = json.loads((root / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    audit.check("saved_validation", saved.get("status") == "pass" and saved.get("failure_count") == 0, {"status": saved.get("status"), "failures": saved.get("failure_count")})
    audit.check("promotion_complete", (root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8").startswith("PASS\n"), (root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8"))
    bytecode = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.suffix == ".pyc" or "__pycache__" in path.parts]
    audit.check("no_bytecode", not bytecode, bytecode)
    return audit.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()
    result = validate(args.record_root.resolve())
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"ED02 ROUND12 VERSION RECORD: {result['status'].upper()}", f"checks={result['check_count']}", f"failures={result['failure_count']}"]
        lines.extend(f"{item['status'].upper()} {item['check']}: {item['detail']}" for item in result["checks"])
        args.text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
