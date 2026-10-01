"""Fail-closed validation for the ED2 slide-3 Round 11 alignment review.

Round 11 is display-only.  It must preserve all Round 10 frames, source crops,
scientific tables, typography, colors, markers, and panel roles while changing
only the requested source geometry and panel-letter positions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND10_ASSEMBLY = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed02_route_policies_and_biased_walk"
    / "assembly_round10_review"
)
ROUND10_REVIEW = ROUND10_ASSEMBLY.parent / "review_2026-08-09_round10"
ROUND11_ROOT = REPO_ROOT / "tmp" / "ed02_round11_alignment_2026-08-09"
REPORT_ROOT = REPO_ROOT / "tmp" / "ed02_round11_validator_2026-08-09"
CURRENT_POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"
PANELS = tuple("abcdefghi")
FORMATS = ("svg", "pdf", "png", "tiff")

EXPECTED_CROPS = {
    "a": (0.273611, 0.168611, 1.413056, 1.724722),
    "b": (0.000289, 0.198611, 2.653868, 1.509722),
    "c": (0.000289, 0.198611, 2.653868, 1.509722),
    "d": (0.255278, 0.0, 1.666667, 2.395833),
    "e": (0.0, 0.120331, 1.333333, 2.317169),
    "f": (0.066944, 0.147007, 4.166667, 2.410534),
    "g": (0.051944, 0.106993, 1.833333, 2.210466),
    "h": (0.0, 0.083652, 1.166667, 2.350514),
    "i": (0.066944, 0.147007, 4.166667, 2.410534),
}
EXPECTED_FRAMES = {
    "a": {"left": 2.0, "top": 20.178340, "width": 148.0, "height": 202.119941},
    "b": {"left": 152.0, "top": 80.0, "width": 282.0, "height": 139.333824},
    "c": {"left": 436.0, "top": 80.0, "width": 282.0, "height": 139.333824},
    "d": {"left": 2.0, "top": 224.298282, "width": 139.476654, "height": 236.761689},
    "e": {"left": 143.476654, "top": 224.298282, "width": 143.698408, "height": 236.761689},
    "f": {"left": 289.175063, "top": 224.298282, "width": 428.824937, "height": 236.761689},
    "g": {"left": 2.0, "top": 497.697222, "width": 171.175063, "height": 202.124437},
    "h": {"left": 175.175063, "top": 482.202860, "width": 112.0, "height": 217.618800},
    "i": {"left": 289.175063, "top": 463.059971, "width": 428.824937, "height": 236.761689},
}
EXPECTED_LABELS = {
    "a": {"left": 4.0, "top": 27.5, "width": 18.0, "height": 16.0},
    "b": {"left": 154.0, "top": 27.5, "width": 18.0, "height": 16.0},
    "c": {"left": 438.0, "top": 27.5, "width": 18.0, "height": 16.0},
    "d": {"left": 4.0, "top": 229.5, "width": 18.0, "height": 16.0},
    "e": {"left": 145.476654, "top": 229.5, "width": 18.0, "height": 16.0},
    "f": {"left": 291.175063, "top": 229.5, "width": 18.0, "height": 16.0},
    "g": {"left": 4.0, "top": 473.5, "width": 18.0, "height": 16.0},
    "h": {"left": 177.175063, "top": 473.5, "width": 18.0, "height": 16.0},
    "i": {"left": 291.175063, "top": 473.5, "width": 18.0, "height": 16.0},
}
LEGEND_FRAME = {"left": 337.0, "top": 57.0, "width": 196.0, "height": 18.0}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, passed: bool, detail: object = "") -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: object, b: object, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def exact_frame(actual: object, expected: dict[str, float], tol: float = 1e-6) -> bool:
    return isinstance(actual, dict) and set(actual) == set(expected) and all(
        close(actual[key], expected[key], tol) for key in expected
    )


def parse_svg(path: Path) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    view = tuple(float(value) for value in root.attrib["viewBox"].split())
    rects: list[tuple[float, float, float, float]] = []
    for node in root.findall(".//{http://www.w3.org/2000/svg}clipPath/{http://www.w3.org/2000/svg}rect"):
        rects.append(tuple(float(node.attrib[key]) for key in ("x", "y", "width", "height")))
    return view, rects


def grid_bbox(path: Path) -> tuple[float, float, float, float]:
    root = ET.parse(path).getroot()
    points: list[tuple[float, float]] = []
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for node in root.iter("{http://www.w3.org/2000/svg}path"):
        if "stroke: #001b2a" not in node.attrib.get("style", ""):
            continue
        values = [float(value) for value in re.findall(number, node.attrib.get("d", ""))]
        points.extend(zip(values[::2], values[1::2]))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def map_y(panel: str, svg_y: float, view_height: float) -> float:
    frame = EXPECTED_FRAMES[panel]
    return frame["top"] + svg_y * frame["height"] / view_height


def panel_stems(panel_root: Path) -> dict[str, str]:
    rows = list(csv.DictReader((panel_root / "PANEL_MANIFEST.csv").open(encoding="utf-8")))
    stems: dict[str, str] = {}
    for row in rows:
        panel = row["panel"]
        stems.setdefault(panel, Path(row["path"]).stem)
    return stems


def audit_source_bundle(audit: Audit, panel_root: Path) -> dict[str, object]:
    manifest_path = panel_root / "SOURCE_MANIFEST.json"
    audit.check("source manifest exists", manifest_path.is_file(), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    audit.check("review round is 11", manifest.get("review_round") == 11, manifest.get("review_round"))
    audit.check("candidate remains review-only", manifest.get("status") == "review_candidate_not_current_version_record")
    layout = manifest.get("layout_contract_round11", {})
    frames = layout.get("panel_frames_720", {}) if isinstance(layout, dict) else {}
    labels = layout.get("panel_label_frames_720", {}) if isinstance(layout, dict) else {}
    audit.check(
        "all Round10 picture frames are unchanged",
        set(frames) == set(PANELS) and all(exact_frame(frames[p], EXPECTED_FRAMES[p]) for p in PANELS),
        frames,
    )
    audit.check(
        "all labels are row-aligned one character upward",
        set(labels) == set(PANELS) and all(exact_frame(labels[p], EXPECTED_LABELS[p]) for p in PANELS),
        labels,
    )
    source_crops = manifest.get("source_crop_contract", {})
    for panel in PANELS:
        actual = source_crops.get(panel, {}).get("bbox_inches_bottom_left", []) if isinstance(source_crops, dict) else []
        audit.check(
            f"panel {panel} preserves Round10 source crop",
            len(actual) == 4 and all(close(a, b) for a, b in zip(actual, EXPECTED_CROPS[panel])),
            actual,
        )
    legend = manifest.get("shared_bc_legend", {})
    audit.check("shared legend frame unchanged", exact_frame(legend.get("frame_720"), LEGEND_FRAME), legend)

    panel_manifest = panel_root / "PANEL_MANIFEST.csv"
    rows = list(csv.DictReader(panel_manifest.open(encoding="utf-8"))) if panel_manifest.is_file() else []
    audit.check("panel manifest has exact 36 entries", len(rows) == 36, len(rows))
    audit.check("panel manifest role inventory is a-i", {row.get("panel") for row in rows} == set(PANELS))
    stems = panel_stems(panel_root) if rows else {}
    for panel in PANELS:
        subset = [row for row in rows if row.get("panel") == panel]
        audit.check(f"panel {panel} has four source formats", {row.get("format") for row in subset} == set(FORMATS))
        for row in subset:
            path = panel_root / row["path"]
            audit.check(f"panel {panel} {row['format']} exists", path.is_file(), path)
            if path.is_file():
                audit.check(f"panel {panel} {row['format']} hash closes", sha256(path) == row["sha256"])
                audit.check(f"panel {panel} {row['format']} size closes", path.stat().st_size == int(row["bytes"]))

    vector_rows = list(csv.DictReader((panel_root / "PPT_VECTOR_MANIFEST.csv").open(encoding="utf-8")))
    audit.check("PowerPoint vector manifest has a-i plus one shared legend", {row["role"] for row in vector_rows} == set(PANELS) | {"bc_shared_legend"} and len(vector_rows) == 10)
    for row in vector_rows:
        path = panel_root / row["path"]
        audit.check(f"outlined vector {row['role']} exists", path.is_file(), path)
        if path.is_file():
            audit.check(f"outlined vector {row['role']} hash closes", sha256(path) == row["sha256"])
            audit.check(f"outlined vector {row['role']} has path text", "<text" not in path.read_text(encoding="utf-8"))

    editable_svgs = [panel_root / "panels" / f"{stems[p]}.svg" for p in PANELS]
    audit.check("editable SVGs retain live text", all("<text" in path.read_text(encoding="utf-8") for path in editable_svgs))
    for panel in PANELS:
        png = panel_root / "panels" / f"{stems[panel]}.png"
        tiff = panel_root / "panels" / f"{stems[panel]}.tiff"
        with Image.open(png) as image:
            audit.check(f"panel {panel} PNG is 600 dpi", all(abs(float(v) - 600) <= 1 for v in image.info.get("dpi", (0, 0))), image.info.get("dpi"))
        with Image.open(tiff) as image:
            audit.check(f"panel {panel} TIFF is 600 dpi", all(abs(float(v) - 600) <= 1 for v in image.info.get("dpi", (0, 0))), image.info.get("dpi"))
    return manifest


def audit_tables(audit: Audit, panel_root: Path) -> None:
    current = panel_root / "plot_ready"
    baseline = ROUND10_ASSEMBLY / "plot_ready"
    base_paths = sorted(baseline.glob("*.csv"))
    audit.check("Round10 baseline has 16 plot-ready tables", len(base_paths) == 16, len(base_paths))
    audit.check("Round11 has exact same 16 plot-ready filenames", {p.name for p in current.glob("*.csv")} == {p.name for p in base_paths})
    for path in base_paths:
        candidate = current / path.name
        audit.check(f"table unchanged: {path.name}", candidate.is_file() and sha256(candidate) == sha256(path))


def white_circle_y(path: Path, strokes: set[str]) -> list[float]:
    root = ET.parse(path).getroot()
    ys: list[float] = []
    for node in root.iter("{http://www.w3.org/2000/svg}use"):
        style = node.attrib.get("style", "")
        if "fill: #ffffff" not in style or not any(f"stroke: {stroke}" in style for stroke in strokes):
            continue
        ys.append(float(node.attrib["y"]))
    return sorted(set(round(value, 6) for value in ys))


def audit_geometry(audit: Audit, panel_root: Path) -> dict[str, object]:
    stems = panel_stems(panel_root)
    paths = {p: panel_root / "panels" / f"{stems[p]}.svg" for p in PANELS}
    views: dict[str, tuple[float, float, float, float]] = {}
    clips: dict[str, list[tuple[float, float, float, float]]] = {}
    for panel in PANELS:
        views[panel], clips[panel] = parse_svg(paths[panel])
    audit.check("b/c SVG axes geometry remains identical", clips["b"] == clips["c"], {"b": clips["b"], "c": clips["c"]})
    audit.check("f/i SVG axes geometry remains identical", clips["f"] == clips["i"], {"f": clips["f"], "i": clips["i"]})

    row2 = {}
    for panel in ("d", "e", "f"):
        rect = max(clips[panel], key=lambda item: item[1] + item[3])
        row2[panel] = map_y(panel, rect[1] + rect[3], views[panel][3])
    row3 = {}
    for panel in ("g", "h", "i"):
        rect = max(clips[panel], key=lambda item: item[1] + item[3])
        row3[panel] = map_y(panel, rect[1] + rect[3], views[panel][3])
    audit.check("d/e/f x-axis baselines align", max(row2.values()) - min(row2.values()) <= 0.25, row2)
    audit.check("g/h/i x-axis baselines align", max(row3.values()) - min(row3.values()) <= 0.25, row3)

    d_top = map_y("d", min(item[1] for item in clips["d"]), views["d"][3])
    e_top = map_y("e", min(item[1] for item in clips["e"]), views["e"][3])
    g_top = map_y("g", min(item[1] for item in clips["g"]), views["g"][3])
    h_top = map_y("h", min(item[1] for item in clips["h"]), views["h"][3])
    audit.check("d/e axes envelopes have equal top and height", abs(d_top - e_top) <= 0.25 and abs((row2["d"] - d_top) - (row2["e"] - e_top)) <= 0.25, {"d": [d_top, row2["d"]], "e": [e_top, row2["e"]]})
    audit.check("g/h axes envelopes have equal top and height", abs(g_top - h_top) <= 0.25 and abs((row3["g"] - g_top) - (row3["h"] - h_top)) <= 0.25, {"g": [g_top, row3["g"]], "h": [h_top, row3["h"]]})

    round10_a = ROUND10_ASSEMBLY / "panels" / "ed02a_four_policy_schematic_round10.svg"
    b10 = grid_bbox(round10_a)
    b11 = grid_bbox(paths["a"])
    ratios = ((b11[2] - b11[0]) / (b10[2] - b10[0]), (b11[3] - b11[1]) / (b10[3] - b10[1]))
    audit.check("panel-a grid is exactly 0.8x", all(abs(value - 0.8) <= 0.002 for value in ratios), ratios)
    text = paths["a"].read_text(encoding="utf-8")
    audit.check("panel-a annotation is one-line preceding turn", text.count(">preceding turn</text>") == 1 and "preceding\nturn" not in text)

    png_a = panel_root / "panels" / f"{stems['a']}.png"
    with Image.open(png_a) as image:
        rgb = np.asarray(image.convert("RGB"))
    dark = (rgb[:, :, 0] < 25) & (rgb[:, :, 1] < 45) & (rgb[:, :, 2] < 55)
    gray = np.all(np.abs(rgb.astype(int) - 167) < 10, axis=2)
    first_grid = int(np.nonzero(dark)[0].min())
    yy = np.indices(gray.shape)[0]
    annotation_y = np.nonzero(gray & (yy < first_grid))[0]
    clearance_px = first_grid - int(annotation_y.max()) - 1 if annotation_y.size else -1
    clearance_pt = clearance_px / 600 * 72
    audit.check("preceding-turn label clears the grid by at least 3.5 pt", clearance_pt >= 3.5, clearance_pt)

    d_y = white_circle_y(paths["d"], {"#68737d"})
    h_y = white_circle_y(paths["h"], {"#69737f", "#d26a3a"})
    d_step = float(np.median(np.diff(d_y))) * EXPECTED_FRAMES["d"]["height"] / views["d"][3]
    h_step = float(np.diff(h_y)[0]) * EXPECTED_FRAMES["h"]["height"] / views["h"][3]
    d_reduction = 1 - d_step / 47.00084
    h_reduction = 1 - h_step / 85.19419
    audit.check("panel-d group spacing reduced at least 18%", d_reduction >= 0.18, {"step": d_step, "reduction": d_reduction})
    audit.check("panel-h group spacing reduced at least 22%", h_reduction >= 0.22, {"step": h_step, "reduction": h_reduction})

    overlaps: dict[str, int] = {}
    for panel in PANELS:
        frame, label = EXPECTED_FRAMES[panel], EXPECTED_LABELS[panel]
        left = max(frame["left"], label["left"])
        top = max(frame["top"], label["top"])
        right = min(frame["left"] + frame["width"], label["left"] + label["width"])
        bottom = min(frame["top"] + frame["height"], label["top"] + label["height"])
        if right <= left or bottom <= top:
            overlaps[panel] = 0
            continue
        png = panel_root / "panels" / f"{stems[panel]}.png"
        with Image.open(png) as image:
            rgb = np.asarray(image.convert("RGB"))
        x0 = max(0, math.floor((left - frame["left"]) / frame["width"] * rgb.shape[1]))
        x1 = min(rgb.shape[1], math.ceil((right - frame["left"]) / frame["width"] * rgb.shape[1]))
        y0 = max(0, math.floor((top - frame["top"]) / frame["height"] * rgb.shape[0]))
        y1 = min(rgb.shape[0], math.ceil((bottom - frame["top"]) / frame["height"] * rgb.shape[0]))
        overlaps[panel] = int(np.count_nonzero(np.any(rgb[y0:y1, x0:x1] < 245, axis=2)))
    audit.check("panel-letter rectangles intersect no source ink", all(value == 0 for value in overlaps.values()), overlaps)
    return {
        "row2_baselines": row2,
        "row3_baselines": row3,
        "d_e_tops": {"d": d_top, "e": e_top},
        "g_h_tops": {"g": g_top, "h": h_top},
        "panel_a_scale": ratios,
        "panel_a_annotation_clearance_pt": clearance_pt,
        "d_spacing_reduction": d_reduction,
        "h_spacing_reduction": h_reduction,
        "label_source_ink_pixels": overlaps,
    }


def pptx_notes(path: Path) -> str:
    with ZipFile(path) as archive:
        parts = [name for name in archive.namelist() if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
        texts: list[str] = []
        for name in parts:
            root = ET.fromstring(archive.read(name))
            texts.extend(
                node.text or ""
                for node in root.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}t")
            )
        return " ".join(texts)


def pptx_svg_hashes(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        return sorted(
            hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("ppt/media/") and name.lower().endswith(".svg")
        )


def audit_decks(audit: Audit, panel_root: Path, deck_root: Path, report_path: Path, cumulative: Path, review: Path, cumulative_render: Path, review_render: Path) -> dict[str, object]:
    audit.check("Round11 build report exists", report_path.is_file(), report_path)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    audit.check("build report uses artifact-tool", report.get("authoringTool") == "@oai/artifact-tool", report.get("authoringTool"))
    audit.check("build report tags review round 11", report.get("reviewRound") == 11, report.get("reviewRound"))
    audit.check("build report frames unchanged", set(report.get("frames", {})) == set(PANELS) and all(exact_frame(report["frames"][p], EXPECTED_FRAMES[p]) for p in PANELS), report.get("frames"))
    audit.check("build report labels exact", set(report.get("labels", {})) == set(PANELS) and all(exact_frame(report["labels"][p], EXPECTED_LABELS[p]) for p in PANELS), report.get("labels"))
    audit.check("build report shared legend unchanged", exact_frame(report.get("sharedLegend", {}).get("frame"), LEGEND_FRAME), report.get("sharedLegend"))

    vector_rows = list(csv.DictReader((panel_root / "PPT_VECTOR_MANIFEST.csv").open(encoding="utf-8")))
    expected_hashes = sorted(row["sha256"] for row in vector_rows)
    for tag, path, expected_slides in (("cumulative", cumulative, 12), ("one-slide", review, 1)):
        audit.check(f"{tag} deck exists", path.is_file(), path)
        if not path.is_file():
            continue
        with ZipFile(path) as archive:
            slide_count = sum(bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name)) for name in archive.namelist())
        audit.check(f"{tag} slide count", slide_count == expected_slides, slide_count)
        actual_hashes = pptx_svg_hashes(path)
        audit.check(f"{tag} embeds exact ten Round11 SVG payloads", actual_hashes == expected_hashes, {"actual": actual_hashes, "expected": expected_hashes})
        notes = pptx_notes(path)
        lower = notes.lower()
        normalized = lower.replace("round-11", "round 11")
        audit.check(f"{tag} notes declare Round11 alignment", "round 11" in normalized and "0.8" in normalized and "x-axis" in normalized and "manual" in normalized and "mask" in normalized, normalized[:1000])
        audit.check(f"{tag} notes contain no transient path", "tmp/" not in lower and "tmp\\" not in lower and not re.search(r"[a-z]:\\", lower))

    audit.check("native cumulative target render exists", cumulative_render.is_file(), cumulative_render)
    audit.check("native one-slide target render exists", review_render.is_file(), review_render)
    native_metrics: dict[str, object] = {}
    if cumulative_render.is_file() and review_render.is_file():
        with Image.open(cumulative_render) as image:
            first = np.asarray(image.convert("RGB"))
        with Image.open(review_render) as image:
            second = np.asarray(image.convert("RGB"))
        same_shape = first.shape == second.shape
        delta = np.abs(first.astype(int) - second.astype(int)) if same_shape else np.array([999])
        changed = int(np.count_nonzero(np.any(delta > 0, axis=2))) if same_shape else -1
        max_delta = int(delta.max())
        audit.check("native cumulative/review target renders agree", same_shape and changed <= 5 and max_delta <= 20, {"changed_pixels": changed, "max_channel_delta": max_delta})
        native_metrics = {"changed_pixels": changed, "max_channel_delta": max_delta}

    render_dir = cumulative_render.parent
    baseline_dir = ROUND10_REVIEW / "powerpoint-render-round10"
    preserved = []
    for number in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        new = render_dir / f"Slide{number}.PNG"
        old = baseline_dir / f"Slide{number}.PNG"
        preserved.append(new.is_file() and old.is_file() and sha256(new) == sha256(old))
    audit.check("all unrelated cumulative slides are native-render identical to Round10", len(preserved) == 11 and all(preserved), preserved)
    return {"native_parity": native_metrics, "build_report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=ROUND11_ROOT / "panels")
    parser.add_argument("--deck-root", type=Path, default=ROUND11_ROOT / "deck")
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--cumulative-pptx", type=Path)
    parser.add_argument("--review-pptx", type=Path)
    parser.add_argument("--cumulative-render", type=Path)
    parser.add_argument("--review-render", type=Path)
    parser.add_argument("--json-out", type=Path, default=REPORT_ROOT / "round11-validation.json")
    parser.add_argument("--text-out", type=Path, default=REPORT_ROOT / "round11-validation.txt")
    parser.add_argument("--panel-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = Audit()
    metrics: dict[str, object] = {}
    try:
        audit_source_bundle(audit, args.panel_root)
        audit_tables(audit, args.panel_root)
        metrics.update(audit_geometry(audit, args.panel_root))
        pointer = CURRENT_POINTER.read_text(encoding="utf-8")
        audit.check("accepted ED2 pointer remains Round3", "extended_data_fig02_2026-08-08_round3_slide_inventory" in pointer)
        if not args.panel_only:
            report = args.build_report or args.deck_root / "round11-build-report.json"
            cumulative = args.cumulative_pptx or args.deck_root / "sup-figs-v2_ed02-slide3_round11_review.pptx"
            review = args.review_pptx or args.deck_root / "ed02-slide3_round11_review.pptx"
            cumulative_render = args.cumulative_render or args.deck_root / "powerpoint-render-round11" / "Slide3.PNG"
            review_render = args.review_render or args.deck_root / "one-slide-powerpoint-render-round11" / "Slide1.PNG"
            metrics.update(audit_decks(audit, args.panel_root, args.deck_root, report, cumulative, review, cumulative_render, review_render))
    except Exception:
        audit.check("validator exception", False, traceback.format_exc())
    failures = [row for row in audit.checks if not row["passed"]]
    report = {
        "status": "pass" if not failures else "fail",
        "check_count": len(audit.checks),
        "failure_count": len(failures),
        "metrics": metrics,
        "checks": audit.checks,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"ED2 Round-11 alignment validation: {report['status'].upper()}",
        f"Checks: {report['check_count']}",
        f"Failures: {report['failure_count']}",
    ]
    lines.extend(f"- {row['name']}: {row['detail']}" for row in failures)
    args.text_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if failures and not args.collect_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
