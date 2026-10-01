#!/usr/bin/env python3
"""Independent read-only validator for the ED2 Round-13 version record."""

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
EXPECTED_RECORD = "extended_data_fig02_2026-08-09_round15_panel_g_labels_vor"
DEFAULT_RECORD_ROOT = REPO_ROOT / "docs" / "figure_version_records" / EXPECTED_RECORD
STEM = "ed02_round13_panel_g_labels_version_record"


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
        failures = [row for row in self.checks if row["status"] == "fail"]
        return {"status": "pass" if not failures else "fail", "check_count": len(self.checks), "failure_count": len(failures), "checks": self.checks}


def has_depth_sequence(words: list[tuple[object, ...]]) -> bool:
    tokens = [str(word[4]) for word in words]
    expected = ["Shallow", "1-2", "Middle", "3-4", "Deep", "5"]
    return any(tokens[index : index + len(expected)] == expected for index in range(len(tokens)))


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
    audit.check("science_unchanged", version.get("scientific_values_changed") is False, version.get("scientific_values_changed"))
    audit.check("behavior_only_mask_na", "not applicable" in version.get("manual_mask_applicability", "").lower(), version.get("manual_mask_applicability"))
    source = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    depth = source.get("panel_g_label_contract_round13", {})
    letter = source.get("panel_g_letter_contract_round13", {})
    audit.check("source_identity", source.get("record") == EXPECTED_RECORD and source.get("status") == "accepted_version_record", {"record": source.get("record"), "status": source.get("status")})
    audit.check("depth_label_contract", depth.get("labels") == ["Shallow\n1-2", "Middle\n3-4", "Deep\n5"] and depth.get("source_values_changed") is False, depth)
    audit.check("g_letter_contract", letter.get("previous_top_720") == 473.5 and letter.get("new_top_720") == 463.0 and letter.get("delta_720") == -10.5 and letter.get("other_panel_letters_changed") is False, letter)
    labels = source["layout_contract_round11"]["panel_label_frames_720"]
    audit.check("g_label_top_463", float(labels["g"]["top"]) == 463.0, labels["g"])
    audit.check("tables_16", len(list((root / "plot_ready").glob("*.csv"))) == 16, len(list((root / "plot_ready").glob("*.csv"))))
    audit.check("panels_36", len(list((root / "panels").glob("*"))) == 36, len(list((root / "panels").glob("*"))))
    audit.check("inputs_13", len(list(csv.DictReader((root / "INPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))) == 13, "input manifest")
    audit.check("provenance_6", len([path for path in (root / "analysis_provenance").rglob("*") if path.is_file()]) == 6, "analysis provenance")
    g_pdf = root / "panels" / "ed02g_depth_pbs_profile_round11.pdf"
    with fitz.open(g_pdf) as document:
        page = document[0]
        audit.check("panel_g_depth_labels", has_depth_sequence(page.get_text("words")), page.get_text())
        audit.check("panel_g_vector", not page.get_images(full=True), len(page.get_images(full=True)))
    g_svg = (root / "panels" / "ed02g_depth_pbs_profile_round11.svg").read_text(encoding="utf-8")
    audit.check("panel_g_editable_svg", all(label in g_svg for label in ("Shallow", "1-2", "Middle", "3-4", "Deep", "5")) and "<image" not in g_svg, "depth labels and no raster")
    composite_manifest = json.loads((root / "ED02_ROUND13_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("composite_manifest", composite_manifest.get("status") == "pass" and composite_manifest.get("review_round") == 13 and composite_manifest.get("panel_order") == list("abcdefghi"), composite_manifest.get("panel_order"))
    with fitz.open(root / f"{STEM}.pdf") as document:
        page = document[0]
        g_words = [word for word in page.get_text("words") if word[4] == "g"]
        audit.check("primary_true_vector", document.page_count == 1 and page.rect.width == 540 and page.rect.height == 540 and not page.get_images(full=True) and len(page.get_drawings()) > 900, {"images": len(page.get_images(full=True)), "drawings": len(page.get_drawings())})
        audit.check("primary_depth_labels", has_depth_sequence(page.get_text("words")), "depth label sequence")
        audit.check("primary_g_letter_position", len(g_words) == 1 and abs(float(g_words[0][1]) - 348.24749755859375) < 0.02, [list(word[:5]) for word in g_words])
    svg = (root / f"{STEM}.svg").read_text(encoding="utf-8")
    audit.check("primary_editable_svg", svg.count("<text") > 100 and "<image" not in svg and "href=\"http" not in svg, {"text": svg.count("<text"), "image": svg.count("<image")})
    for suffix in ("png", "tiff"):
        with Image.open(root / f"{STEM}_600dpi.{suffix}") as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"primary_{suffix}_600dpi", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})
    for pptx in ("sup-figs-v2_ed02-slide3_round13_review.pptx", "ed02-slide3_round13_review.pptx"):
        with zipfile.ZipFile(root / "review_evidence" / pptx) as archive:
            vectors = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
        audit.check(f"review_vectors::{pptx}", len(vectors) == 10, len(vectors))
    saved = json.loads((root / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    audit.check("saved_validation", saved.get("status") == "pass" and saved.get("failure_count") == 0, {"status": saved.get("status"), "failures": saved.get("failure_count")})
    audit.check("promotion_complete", (root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8").startswith("PASS\n"), "PASS")
    authority = "\n".join((root / name).read_text(encoding="utf-8") for name in ("SOURCE_MANIFEST.json", "RENDER_RESULT.json", "ED02_VERSION_RECORD_MANIFEST.json"))
    audit.check("portable_authority", "\\tmp\\" not in authority.lower() and "/tmp/" not in authority.lower() and "D:\\" not in authority, "record-local")
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
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if args.text_output:
        lines = [f"ED02 ROUND13 VERSION RECORD: {str(result['status']).upper()}", f"checks={result['check_count']}", f"failures={result['failure_count']}"]
        lines.extend(f"{str(row['status']).upper()} {row['check']}: {row['detail']}" for row in result["checks"])
        args.text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
