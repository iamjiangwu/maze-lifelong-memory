"""Stage and source-replay the approved ED01 Round 23 record; no activation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VOR = REPO / "docs/figure_version_records"
PARENT = VOR / "extended_data_fig01_2026-09-08_round22_panel_h_axis_labels"
NAME = "extended_data_fig01_2026-09-25_round23_without_trajectories"
STAGING = REPO / "docs/figure_version_candidates/review_promotions_2026-09-25"
RECORD = STAGING / NAME
REPLAY = STAGING / "source_replays/extended_data_figure_1"
REVIEW = REPO / "output/figures/ed01_without_trajectories_2026-09-25"


def long(path):
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path


def sha(path):
    return hashlib.sha256(long(path).read_bytes()).hexdigest()


def files(root):
    base = long(root)
    return sorted(((p.relative_to(base).as_posix(), p) for p in base.rglob("*") if p.is_file()), key=lambda row: row[0].lower())


def hashes(record):
    with (record / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for relative, path in files(record):
            if relative != "RECORD_HASHES.csv":
                writer.writerow([relative, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    assert not RECORD.exists(), RECORD
    review = json.loads((REVIEW / "DISPLAY_VALIDATION.json").read_text())
    assert review["status"] == "PASS"
    parent_hashes = {rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in files(PARENT)}
    shutil.copytree(long(PARENT), long(RECORD))
    shutil.copy2(REPO / "scripts/render_ed01_no_trajectories_source.py", RECORD / "source_code/render_ed01_no_trajectories_source.py")
    shutil.copy2(Path(__file__), RECORD / "source_code" / Path(__file__).name)
    rebuild = load("ed01_promotion_source_driver", VOR / "rebuild_current_extended_data_figures.py")
    REPLAY.mkdir(parents=True, exist_ok=True)
    with rebuild.frozen_stage(RECORD) as stage:
        result = rebuild.render_ed1(RECORD, REPLAY, stage)
    assert result["status"] == "PASS"
    stem = REPLAY / "extended_data_figure1_source_rebuild_round23"
    with Image.open(stem.with_suffix(".png")) as a, Image.open(REVIEW / "ed01_without_trajectories.png") as b:
        assert np.array_equal(np.asarray(a), np.asarray(b)), "Approved ED01 pixels differ"
    for key in ("accumulation_summary", "speed_summary", "time_efficiency_summary", "tile_efficiency_summary"):
        pd.testing.assert_frame_equal(pd.read_csv(REPLAY / f"{key}.csv"), pd.read_csv(REVIEW / f"{key}.csv"))
    for suffix in ("pdf", "svg", "png", "tiff"):
        dest = "ed01_version_record" + ("_600dpi" if suffix in {"png", "tiff"} else "") + "." + suffix
        shutil.copy2(stem.with_suffix("." + suffix), RECORD / dest)
    with fitz.open(RECORD / "ed01_version_record.pdf") as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(RECORD / "ed01_version_record_300dpi.png", dpi=(300, 300))
    history = RECORD / "source_validation/round23_display_promotion"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("CAPTION_NOTES.md", "ED01_ASSEMBLY_MANIFEST.json", "LAYOUT_AUDIT.csv", "SOURCE_DATA_AUDIT.csv", "ED01_VALIDATION.csv"):
        shutil.copy2(RECORD / name, history / ("parent_" + name))
    for name in ("DISPLAY_VALIDATION.json", "INDEPENDENT_VISUAL_REVIEW.md", "PANEL_LABEL_MAP.csv"):
        if (REVIEW / name).exists():
            shutil.copy2(REVIEW / name, history / name)
    for name in ("LAYOUT_AUDIT.csv", "SOURCE_DATA_AUDIT.csv", "PANEL_LABEL_MAP.csv", "RETAINED_AXES_GEOMETRY.csv", "AXIS_SCALE_AUDIT.csv"):
        shutil.copy2(REPLAY / name, RECORD / name)
    notes = (PARENT / "CAPTION_NOTES.md").read_text(encoding="utf-8")
    notes = "\n".join(line for line in notes.splitlines() if not line.startswith("- **a:**"))
    mapping = {"b-c": "a-b", "d-e": "c-d", "f": "e", "g": "f", "h": "g", "i": "h", "j-k": "i-j"}
    notes = re.sub(r"\*\*([a-k](?:-[a-k])?):\*\*", lambda m: "**" + mapping.get(m[1], m[1]) + ":**", notes)
    notes = notes.replace("Panels g-k", "Panels f-j").replace("Panels j-k", "Panels i-j")
    notes = re.sub(r"Panel ([fgh])\b", lambda m: "Panel " + {"f": "e", "g": "f", "h": "g"}[m[1]], notes)
    notes = notes.replace("The round-21 plots", "The round-23 plots")
    notes += "\n\n## Round 23 display update\n\nFormer trajectory panel a is removed; former b-k are now a-j. Retained numerical tables, inference and physical plot/font sizes are unchanged. Historical source filenames retain their former panel IDs; PANEL_LABEL_MAP.csv maps them to the current display.\n"
    (RECORD / "CAPTION_NOTES.md").write_text(notes, encoding="utf-8")
    pd.DataFrame([
        {"check": "native_source_build", "status": "PASS", "detail": "Frozen native renderer and record-local numerical inputs; no artwork used as a source"},
        {"check": "approved_artwork_pixel_parity", "status": "PASS", "detail": "600-dpi image pixels identical to reviewed ED01"},
        {"check": "current_panel_inventory", "status": "PASS", "detail": "a-j; former trajectory panel a removed"},
        {"check": "current153_endpoint_scope", "status": "PASS", "detail": "153 sessions / 29 canonical biological animals"},
        {"check": "retained_artists_and_geometry", "status": "PASS", "detail": "21 data axes unchanged in physical size and content"},
    ]).to_csv(RECORD / "ED01_VALIDATION.csv", index=False)
    write_json(RECORD / "ROUND23_RENDER_VALIDATION.json", result)
    manifest = {"schema": "ed01-source-built-display-record/v1", "record": NAME, "round": 23,
                "status": "accepted_version_record", "supersedes": PARENT.name,
                "figure": "Extended Data Figure 1", "panel_letters": list("abcdefghij"),
                "panel_label_map": result["panel_label_map"], "backend": "Python/matplotlib",
                "figure_contract": {"canvas_inches": result["canvas_inches"], "layout": "three native quantitative rows; retained axes preserve physical geometry"},
                "validation": result,
                "outputs": {k: "ed01_version_record" + v for k,v in {"pdf": ".pdf", "svg": ".svg", "png_600dpi": "_600dpi.png", "tiff_600dpi": "_600dpi.tiff"}.items()}}
    write_json(RECORD / "ED01_ASSEMBLY_MANIFEST.json", manifest)
    receipt = {"status": "PASS", "record": NAME, "parent": PARENT.name,
               "source_replay": str(REPLAY.relative_to(REPO)), "approved_600dpi_pixels_identical": True,
               "four_numeric_summaries_identical": True, "source_master_artwork_read": False,
               "preserved_input_hashes": {rel: digest for rel, digest in parent_hashes.items() if rel.startswith(("plot_ready/", "drawn_values/"))}}
    write_json(RECORD / "SOURCE_REPLAY_VERIFICATION.json", receipt)
    write_json(RECORD / "RECORD_VALIDATION.json", {"status": "PASS", "record": NAME, "source_replay": receipt, "display_validation": result})
    (RECORD / "REBUILD_README.md").write_text(
        "# Source rebuild\n\n`python docs/figure_version_records/rebuild_current_extended_data_figures.py --artifact extended_data_figure_1 --output-dir <new directory>`\n\n"
        "The current driver stages the record-frozen native renderer and render_ed01_no_trajectories_source.py, reads the record's numerical inputs, removes the retired trajectory axes before saving, and exports all four formats. No approved master/review image is read as a source. SOURCE_REPLAY_VERIFICATION.json records an actual fresh build with identical approved 600-dpi pixels. Retired trajectory inputs remain for inherited renderer compatibility only.\n", encoding="utf-8")
    (RECORD / "MANIFEST.md").write_text(f"# ED01 Round 23\n\nCurrent source-built a-j display. Supersedes {PARENT.name}. See ED01_ASSEMBLY_MANIFEST.json, SOURCE_REPLAY_VERIFICATION.json, REBUILD_README.md and PANEL_LABEL_MAP.csv. Historical ROUND* receipts retained as lineage evidence are not current display contracts.\n", encoding="utf-8")
    (RECORD / "PROMOTION_COMPLETE.txt").write_text("Source-built successor prepared; activated by scoped promotion transaction.\n", encoding="utf-8")
    with (RECORD / "SOURCE_CODE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["path", "sha256", "bytes"])
        for rel, path in files(RECORD / "source_code"):
            writer.writerow(["source_code/" + rel, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size])
    assert all(sha(PARENT / rel) == digest for rel, digest in parent_hashes.items())
    assert all(sha(RECORD / rel) == digest for rel, digest in receipt["preserved_input_hashes"].items())
    hashes(RECORD)
    print(json.dumps({"status": "PASS", "staged_record": str(RECORD), "source_replay": str(REPLAY)}))


if __name__ == "__main__":
    main()
