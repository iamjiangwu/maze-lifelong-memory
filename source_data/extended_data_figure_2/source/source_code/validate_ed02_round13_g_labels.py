#!/usr/bin/env python3
"""Fail-closed validation for the ED2 Round-13 panel-g label correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import fitz
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "tmp" / "ed02_round13_g_labels_2026-08-09"
SOURCE = ROOT / "panels"
COMPOSITE = ROOT / "composite"
DECK = ROOT / "deck"
BASELINE = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor"
)
STEM = "ed02_round13_panel_g_labels_version_record"
DEPTH_LINES = ("Shallow", "1-2", "Middle", "3-4", "Deep", "5")
G_PANEL = "ed02g_depth_pbs_profile_round11"


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
        self.checks.append(
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    def result(self) -> dict[str, object]:
        failures = [row for row in self.checks if row["status"] == "fail"]
        return {
            "status": "pass" if not failures else "fail",
            "check_count": len(self.checks),
            "failure_count": len(failures),
            "checks": self.checks,
        }


def pixel_difference(first: Path, second: Path) -> tuple[int, int]:
    with Image.open(first).convert("RGB") as a, Image.open(second).convert("RGB") as b:
        if a.size != b.size:
            return -1, 255
        difference = ImageChops.difference(a, b)
        pixels = list(difference.getdata())
        return sum(pixel != (0, 0, 0) for pixel in pixels), max(max(pixel) for pixel in pixels)


def has_depth_sequence(words: list[tuple[object, ...]]) -> bool:
    tokens = [str(word[4]) for word in words]
    expected = ["Shallow", "1-2", "Middle", "3-4", "Deep", "5"]
    return any(tokens[index : index + len(expected)] == expected for index in range(len(tokens)))


def validate() -> dict[str, object]:
    audit = Audit()
    for path in (SOURCE, COMPOSITE, DECK, BASELINE):
        audit.check(f"exists::{path.name}", path.is_dir(), str(path))
    if any(not path.is_dir() for path in (SOURCE, COMPOSITE, DECK, BASELINE)):
        return audit.result()

    manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    g_contract = manifest.get("panel_g_label_contract_round13", {})
    letter_contract = manifest.get("panel_g_letter_contract_round13", {})
    audit.check(
        "panel_g_depth_contract",
        g_contract.get("labels") == ["Shallow\n1-2", "Middle\n3-4", "Deep\n5"]
        and g_contract.get("group_positions") == [0.28, 1.0, 1.72]
        and g_contract.get("source_values_changed") is False,
        g_contract,
    )
    audit.check(
        "panel_g_letter_exact_raise",
        letter_contract.get("previous_top_720") == 473.5
        and letter_contract.get("new_top_720") == 463.0
        and letter_contract.get("delta_720") == -10.5
        and letter_contract.get("other_panel_letters_changed") is False,
        letter_contract,
    )
    label_frames = manifest["layout_contract_round11"]["panel_label_frames_720"]
    baseline_manifest = json.loads((BASELINE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    baseline_labels = baseline_manifest["layout_contract_round11"]["panel_label_frames_720"]
    audit.check("source_label_g_top", float(label_frames["g"]["top"]) == 463.0, label_frames["g"])
    for panel in "abcdefhi":
        audit.check(
            f"other_label_unchanged::{panel}",
            label_frames[panel] == baseline_labels[panel],
            label_frames[panel],
        )

    candidate_tables = sorted((SOURCE / "plot_ready").glob("*.csv"))
    baseline_tables = sorted((BASELINE / "plot_ready").glob("*.csv"))
    audit.check("table_inventory_16", len(candidate_tables) == len(baseline_tables) == 16, len(candidate_tables))
    for path in candidate_tables:
        baseline = BASELINE / "plot_ready" / path.name
        audit.check(
            f"table_byte_parity::{path.name}",
            baseline.is_file() and sha256(path) == sha256(baseline),
            sha256(path),
        )

    panels = sorted((SOURCE / "panels").glob("*"))
    audit.check(
        "panel_exports_36",
        len(panels) == 36 and {path.suffix.lower() for path in panels} == {".svg", ".pdf", ".png", ".tiff"},
        len(panels),
    )
    for panel in "abcdefhi":
        for suffix in (".png", ".tiff"):
            candidate = next((SOURCE / "panels").glob(f"ed02{panel}*{suffix}"))
            baseline = BASELINE / "panels" / candidate.name
            audit.check(
                f"unchanged_raster::{candidate.name}",
                baseline.is_file() and sha256(candidate) == sha256(baseline),
                sha256(candidate),
            )

    g_pdf = SOURCE / "panels" / f"{G_PANEL}.pdf"
    with fitz.open(g_pdf) as document:
        page = document[0]
        text = page.get_text()
        words = page.get_text("words")
        labels = [word for word in words if word[4] in DEPTH_LINES]
        audit.check(
            "panel_g_pdf_depth_labels_once",
            has_depth_sequence(words) and len(labels) == 6,
            [str(word[4]) for word in labels],
        )
        audit.check(
            "panel_g_pdf_depth_labels_inside",
            all(
                word[0] >= 0.5
                and word[1] >= 0.5
                and word[2] <= page.rect.width - 0.5
                and word[3] <= page.rect.height - 0.5
                for word in labels
            ),
            [list(word[:5]) for word in labels],
        )
        audit.check("panel_g_pdf_true_vector", not page.get_images(full=True), len(page.get_images(full=True)))
    g_svg_text = (SOURCE / "panels" / f"{G_PANEL}.svg").read_text(encoding="utf-8")
    audit.check(
        "panel_g_editable_svg_labels",
        all(label in g_svg_text for label in DEPTH_LINES) and "<image" not in g_svg_text,
        {label: label in g_svg_text for label in DEPTH_LINES},
    )
    for suffix in (".png", ".tiff"):
        path = SOURCE / "panels" / f"{G_PANEL}{suffix}"
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(
                f"panel_g_600dpi::{suffix}",
                min(dpi) >= 599.0,
                {"size": image.size, "dpi": dpi},
            )

    composite_manifest = json.loads(
        (COMPOSITE / "ED02_ROUND13_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit.check(
        "composite_manifest_pass",
        composite_manifest.get("status") == "pass"
        and composite_manifest.get("review_round") == 13
        and composite_manifest.get("panel_order") == list("abcdefghi"),
        composite_manifest.get("panel_order"),
    )
    placement = next(row for row in composite_manifest["placements"] if row["role"] == "g")
    audit.check(
        "composite_g_label_frame",
        float(placement["label_frame_720"]["top"]) == 463.0,
        placement["label_frame_720"],
    )
    pdf = COMPOSITE / f"{STEM}.pdf"
    with fitz.open(pdf) as document:
        page = document[0]
        text = page.get_text()
        g_words = [word for word in page.get_text("words") if word[4] == "g"]
        audit.check(
            "composite_true_vector",
            document.page_count == 1
            and page.rect.width == 540
            and page.rect.height == 540
            and not page.get_images(full=True)
            and len(page.get_drawings()) > 900,
            {"images": len(page.get_images(full=True)), "drawings": len(page.get_drawings())},
        )
        audit.check(
            "composite_depth_labels",
            has_depth_sequence(page.get_text("words")),
            {label: text.count(label) for label in DEPTH_LINES},
        )
        audit.check(
            "composite_g_letter_position",
            len(g_words) == 1 and abs(float(g_words[0][1]) - 348.24749755859375) < 0.02,
            [list(word[:5]) for word in g_words],
        )
    svg = COMPOSITE / f"{STEM}.svg"
    svg_text = svg.read_text(encoding="utf-8")
    audit.check(
        "composite_editable_svg",
        svg_text.count("<text") > 100 and "<image" not in svg_text and "href=\"http" not in svg_text,
        {"text": svg_text.count("<text"), "image": svg_text.count("<image")},
    )
    for suffix in ("png", "tiff"):
        path = COMPOSITE / f"{STEM}_600dpi.{suffix}"
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(
                f"composite_{suffix}_600dpi",
                image.size == (4500, 4500) and min(dpi) >= 599.0,
                {"size": image.size, "dpi": dpi},
            )

    build = json.loads((DECK / "round13-build-report.json").read_text(encoding="utf-8"))
    audit.check("deck_build_pass", build.get("status") == "pass" and len(build.get("results", [])) == 2, build.get("status"))
    for result in build.get("results", []):
        audit.check(
            f"deck_g_label_delta::{result['tag']}",
            abs(float(result["panelGLabel"]["newFrame"]["top"]) - 463.0) < 1e-6
            and abs(float(result["panelGLabel"]["deltaTop"]) + 10.5) < 0.001,
            result["panelGLabel"],
        )
        pptx = DECK / result["output"]
        with zipfile.ZipFile(pptx) as archive:
            vectors = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
            notes = "\n".join(
                archive.read(name).decode("utf-8", errors="replace")
                for name in archive.namelist()
                if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")
            )
        audit.check(f"deck_vector_inventory::{result['tag']}", len(vectors) == 10, len(vectors))
        audit.check(
            f"deck_notes_contract::{result['tag']}",
            all(label in notes for label in ("Shallow 1-2", "Middle 3-4", "Deep 5"))
            and "10.5-pt" in notes
            and "tmp" not in notes.lower(),
            "durable panel-g correction notes",
        )
        inspect = DECK / f"{result['tag']}-after-inspect.ndjson"
        records = [json.loads(line) for line in inspect.read_text(encoding="utf-8").splitlines() if line.strip()]
        slide_number = int(result["editedSlide"])
        images = [row for row in records if row.get("kind") == "image" and row.get("slide") == slide_number]
        label_rows = [row for row in records if row.get("kind") == "textbox" and row.get("slide") == slide_number and row.get("name") == "ED2 panel label g"]
        audit.check(f"deck_images_10::{result['tag']}", len(images) == 10, len(images))
        audit.check(
            f"deck_label_g_frame::{result['tag']}",
            len(label_rows) == 1 and abs(float(label_rows[0]["bbox"][1]) - 463.0) < 0.01,
            label_rows[0]["bbox"] if label_rows else None,
        )

    cumulative = DECK / "powerpoint-render-round13" / "Slide3.PNG"
    review = DECK / "one-slide-powerpoint-render-round13" / "Slide1.PNG"
    diff_pixels, max_delta = pixel_difference(cumulative, review)
    audit.check(
        "native_target_parity",
        0 <= diff_pixels <= 5 and max_delta <= 20,
        {"different_pixels": diff_pixels, "max_delta": max_delta},
    )
    for slide in tuple(range(1, 3)) + tuple(range(4, 13)):
        candidate = DECK / "powerpoint-render-round13" / f"Slide{slide}.PNG"
        baseline = DECK / "baseline-powerpoint-render-round12" / f"Slide{slide}.PNG"
        audit.check(
            f"unrelated_slide_pixel_identity::{slide}",
            candidate.is_file() and baseline.is_file() and sha256(candidate) == sha256(baseline),
            sha256(candidate) if candidate.is_file() else "missing",
        )
    bytecode = [path for path in ROOT.rglob("*") if path.suffix == ".pyc" or "__pycache__" in path.parts]
    audit.check("candidate_no_bytecode", not bytecode, [path.as_posix() for path in bytecode])
    return audit.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=ROOT / "ROUND13_FIX_VALIDATION.json")
    parser.add_argument("--text-output", type=Path, default=ROOT / "ROUND13_FIX_VALIDATION.txt")
    args = parser.parse_args()
    result = validate()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    args.json_output.write_text(rendered + "\n", encoding="utf-8")
    lines = [
        f"ED02 ROUND13 PANEL-G LABEL FIX: {str(result['status']).upper()}",
        f"checks={result['check_count']}",
        f"failures={result['failure_count']}",
    ]
    lines.extend(
        f"{str(row['status']).upper()} {row['check']}: {row['detail']}"
        for row in result["checks"]
    )
    args.text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
