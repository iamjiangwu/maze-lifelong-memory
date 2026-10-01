"""Build and validate ED01 Round 24's narrower native canvas; no activation."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VOR = REPO / "docs/figure_version_records"
PARENT = VOR / "extended_data_fig01_2026-09-25_round23_without_trajectories"
NAME = "extended_data_fig01_2026-09-25_round24_trim_right_margin"
STAGE_ROOT = REPO / "docs/figure_version_candidates/review_promotions_2026-09-25"
RECORD = STAGE_ROOT / NAME
REPLAY = STAGE_ROOT / "source_replays/ed01_round24_trim_right_margin_native_bbox"


def long(path):
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def sha(path):
    return hashlib.sha256(long(path).read_bytes()).hexdigest()


def files(root):
    base = long(root)
    return sorted(((p.relative_to(base).as_posix(), p) for p in base.rglob("*") if p.is_file()), key=lambda row: row[0].lower())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def normalized(value):
    if isinstance(value, (fitz.Point, fitz.Rect, fitz.Quad)):
        return normalized(list(value))
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return tuple(normalized(x) for x in value)
    if isinstance(value, dict):
        return tuple((key, normalized(child)) for key, child in sorted(value.items()) if key not in {"seqno", "layer"})
    return value


def main():
    assert not REPLAY.exists(), "Refusing to overwrite a source replay"
    parent_hashes = {relative: sha(path) for relative, path in files(PARENT)}
    if RECORD.exists():
        assert not (RECORD / "ROUND24_RENDER_VALIDATION.json").exists(), "Candidate is already sealed"
    else:
        shutil.copytree(long(PARENT), long(RECORD))
    shutil.copy2(REPO / "scripts/render_ed01_no_trajectories_source.py", RECORD / "source_code/render_ed01_no_trajectories_source.py")
    shutil.copy2(Path(__file__), RECORD / "source_code" / Path(__file__).name)
    rebuild = load("ed01_round24_driver", VOR / "rebuild_current_extended_data_figures.py")
    REPLAY.mkdir(parents=True)
    with rebuild.frozen_stage(RECORD) as stage:
        result = rebuild.render_ed1(RECORD, REPLAY, stage)
    assert result["status"] == "PASS"
    stem = REPLAY / "extended_data_figure1_source_rebuild_round24"
    parent_validation = json.loads((PARENT / "ROUND23_RENDER_VALIDATION.json").read_text())
    assert result["canvas_inches"][1] == parent_validation["canvas_inches"][1]
    assert result["canvas_inches"][0] < parent_validation["canvas_inches"][0]

    old_pixels = np.asarray(Image.open(PARENT / "ed01_version_record_600dpi.png"))
    new_pixels = np.asarray(Image.open(stem.with_suffix(".png")))
    assert old_pixels.shape[0] == new_pixels.shape[0]
    comparison = old_pixels[:, :new_pixels.shape[1], :]
    raster_exact = np.array_equal(new_pixels, comparison)
    if not raster_exact:
        changed = np.any(new_pixels != comparison, axis=2)
        yy, xx = np.where(changed)
        raise AssertionError(f"Retained pixels changed: {changed.sum()}; bbox={(xx.min(), yy.min(), xx.max(), yy.max())}; maxabs={np.abs(new_pixels.astype(int)-comparison.astype(int)).max()}")
    assert np.all(old_pixels[:, new_pixels.shape[1]:, :3] == 255), "Removed band contained visible pixels"
    old_geometry = pd.read_csv(PARENT / "RETAINED_AXES_GEOMETRY.csv")
    new_geometry = pd.read_csv(REPLAY / "RETAINED_AXES_GEOMETRY.csv")
    pd.testing.assert_frame_equal(old_geometry, new_geometry, rtol=0, atol=1e-12)

    signature_helper = load("ed01_round24_svg_signature", REPO / "scripts/render_figure2_labels_stars_revision.py")
    signatures = []
    for path in (PARENT / "ed01_version_record.svg", stem.with_suffix(".svg")):
        root = ET.parse(path).getroot()
        assert not list(root.iter("{http://www.w3.org/2000/svg}image"))
        root.set("width", "CANVAS_WIDTH")
        root.set("viewBox", "CANVAS_VIEWBOX")
        for date in root.iter("{http://purl.org/dc/elements/1.1/}date"):
            date.text = "GENERATED_DATE"
        background = root.find(".//*[@id='patch_1']/{http://www.w3.org/2000/svg}path")
        assert background is not None
        background.set("d", "CANVAS_BACKGROUND_RECTANGLE")
        signatures.append(signature_helper.svg_signature(root))
    assert signatures[0] == signatures[1], "SVG contains a change beyond the blank canvas/background"
    drawings, spans, sizes = [], [], []
    for path in (PARENT / "ed01_version_record.pdf", stem.with_suffix(".pdf")):
        with fitz.open(path) as doc:
            page = doc[0]
            assert len(doc) == 1 and not page.get_images(full=True)
            sizes.append([float(page.rect.width), float(page.rect.height)])
            content = []
            for drawing in page.get_drawings():
                if drawing.get("fill") == (1.0, 1.0, 1.0) and max(abs(float(a) - float(b)) for a, b in zip(drawing["rect"], page.rect)) < .001:
                    continue
                content.append(normalized(drawing))
            drawings.append(Counter(content))
            spans.append([normalized({key: span[key] for key in ("text", "origin", "bbox", "font", "size", "color")})
                          for block in page.get_text("dict")["blocks"] if "lines" in block
                          for line in block["lines"] for span in line["spans"]])
    assert drawings[0] == drawings[1], "PDF vector drawing geometry changed"
    assert spans[0] == spans[1], "PDF text positions or styles changed"
    assert sizes[0][1] == sizes[1][1] and sizes[0][0] > sizes[1][0]
    for suffix in ("pdf", "svg", "png", "tiff"):
        name = "ed01_version_record" + ("_600dpi" if suffix in {"png", "tiff"} else "") + "." + suffix
        shutil.copy2(stem.with_suffix("." + suffix), RECORD / name)
    with fitz.open(RECORD / "ed01_version_record.pdf") as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(RECORD / "ed01_version_record_300dpi.png", dpi=(300, 300))
    history = RECORD / "source_validation/round24_right_margin"
    history.mkdir(parents=True)
    for name in ("ED01_ASSEMBLY_MANIFEST.json", "LAYOUT_AUDIT.csv", "ED01_VALIDATION.csv", "SOURCE_REPLAY_VERIFICATION.json", "RECORD_VALIDATION.json"):
        shutil.copy2(RECORD / name, history / ("parent_" + name))
    shutil.copy2(REPLAY / "ed01_source_preview.png", history / "ed01_trimmed_right_margin_preview.png")
    for name in ("LAYOUT_AUDIT.csv", "SOURCE_DATA_AUDIT.csv", "PANEL_LABEL_MAP.csv", "RETAINED_AXES_GEOMETRY.csv", "AXIS_SCALE_AUDIT.csv"):
        shutil.copy2(REPLAY / name, RECORD / name)
    dump(RECORD / "ROUND24_RENDER_VALIDATION.json", result)
    manifest = json.loads((PARENT / "ED01_ASSEMBLY_MANIFEST.json").read_text())
    manifest.update(record=NAME, round=24, supersedes=PARENT.name, validation=result)
    manifest["figure_contract"]["canvas_inches"] = result["canvas_inches"]
    manifest["figure_contract"]["layout"] += "; right blank canvas trimmed to 4-point safety margin without moving or scaling retained content"
    dump(RECORD / "ED01_ASSEMBLY_MANIFEST.json", manifest)
    caption = (PARENT / "CAPTION_NOTES.md").read_text(encoding="utf-8")
    caption += "\n## Round 24 canvas update\n\nOnly blank space at the right edge is removed. All retained panel coordinates, fonts, numerical values, annotations, and panel IDs are unchanged.\n"
    (RECORD / "CAPTION_NOTES.md").write_text(caption, encoding="utf-8")
    pd.DataFrame([
        {"check": "native_source_build", "status": "PASS", "detail": "Native source canvas resized; accepted artwork is comparison-only"},
        {"check": "retained_region_pixel_parity", "status": "PASS", "detail": "All retained 600-dpi pixels are exactly parent pixels; removed band entirely white"},
        {"check": "vector_geometry_text_parity", "status": "PASS", "detail": "PDF drawings/text and all non-canvas SVG nodes unchanged"},
        {"check": "current_panel_inventory", "status": "PASS", "detail": "a-j; 21 retained data axes"},
        {"check": "current153_endpoint_scope", "status": "PASS", "detail": "153 sessions / 29 canonical biological animals"},
        {"check": "right_safety_margin", "status": "PASS", "detail": "4 points beyond measured native artist extent"},
    ]).to_csv(RECORD / "ED01_VALIDATION.csv", index=False)
    input_hashes = {rel: digest for rel, digest in parent_hashes.items() if rel.startswith(("plot_ready/", "drawn_values/"))}
    assert all(sha(PARENT / rel) == digest for rel, digest in parent_hashes.items())
    assert all(sha(RECORD / rel) == digest for rel, digest in input_hashes.items())
    receipt = {"status": "PASS", "record": NAME, "parent": PARENT.name,
               "source_replay": REPLAY.relative_to(REPO).as_posix(), "source_master_artwork_read": False,
               "native_source_canvas_resize_only": True, "retained_600dpi_pixels_identical": True,
               "removed_600dpi_band_all_white": True, "parent_png_shape": list(old_pixels.shape), "new_png_shape": list(new_pixels.shape),
               "all_pdf_text_spans_unchanged": len(spans[1]), "all_pdf_drawings_unchanged": sum(drawings[1].values()),
               "noncanvas_svg_nodes_unchanged": len(signatures[1]), "pdf_page_sizes_points": sizes,
               "scientific_tables_unchanged_count": len(input_hashes), "preserved_input_hashes": input_hashes,
               "right_margin_points": result["right_margin_points"], "canvas_inches": result["canvas_inches"],
               "native_renderer_sha256": sha(RECORD / "source_code/render_ed01_no_trajectories_source.py")}
    dump(RECORD / "SOURCE_REPLAY_VERIFICATION.json", receipt)
    dump(history / "DISPLAY_VALIDATION.json", receipt)
    dump(RECORD / "RECORD_VALIDATION.json", {"status": "PASS", "record": NAME, "source_replay": receipt, "display_validation": result})
    (RECORD / "REBUILD_README.md").write_text("# Source rebuild\n\n`python docs/figure_version_records/rebuild_current_extended_data_figures.py --artifact extended_data_figure_1 --output-dir <new directory>`\n\nThe current driver stages the record-frozen native renderer and `render_ed01_no_trajectories_source.py`, reads the record-local numerical tables, removes retired trajectory axes, and fits the native export viewport to the retained content with 4-point upper/right margins. Original logical x transforms remain unchanged, preserving all physical positions, plot sizes, typography and legends. The narrower source-rendered export is not a crop of finished artwork. No approved artwork is a source. `SOURCE_REPLAY_VERIFICATION.json` proves exact retained-region pixel parity and native SVG/PDF geometry parity against Round 23. Historical ROUND23 and earlier receipts remain provenance, not current display contracts.\n", encoding="utf-8")
    (RECORD / "MANIFEST.md").write_text(f"# ED01 Round 24\n\nSupersedes `{PARENT.name}`. Removes only entirely blank right-side canvas space using native source figure dimensions. The a-j panels and all data/physical geometry are preserved. See `ROUND24_RENDER_VALIDATION.json`, `SOURCE_REPLAY_VERIFICATION.json` and `REBUILD_README.md`. Historical receipts remain unchanged.\n", encoding="utf-8")
    (RECORD / "PROMOTION_COMPLETE.txt").write_text("Source-built successor prepared and validated; activated by coordinator's scoped promotion transaction.\n", encoding="utf-8")
    with (RECORD / "SOURCE_CODE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for rel, path in files(RECORD / "source_code"):
            writer.writerow(["source_code/" + rel, sha(path), path.stat().st_size])
    with (RECORD / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for rel, path in files(RECORD):
            if rel != "RECORD_HASHES.csv":
                writer.writerow([rel, sha(path), path.stat().st_size])
    print(json.dumps({key: value for key, value in receipt.items() if key != "preserved_input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
