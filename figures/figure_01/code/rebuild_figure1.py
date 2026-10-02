"""Rebuild the approved ten-panel Figure 1 using only record-local inputs.

The composite and individual panels are generated from native plotting artists.
Accepted artwork is never an input. The sole supplied bitmap is the declared
physical-maze illustration; quantitative panels use frozen numerical tables.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import fitz
import numpy as np
import pandas as pd
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inputs(record):
    rows = list(csv.DictReader((record / "SOURCE_INPUTS.csv").open(encoding="utf-8", newline="")))
    for row in rows:
        path = (record / row["path"]).resolve()
        path.relative_to(record.resolve())
        assert sha(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), path
    return rows


def export_individual_panels(builder, runtime, native, image, bounds, output):
    """Export isolated native source artists, retaining their physical sizes."""
    folder = output / "figure1_individual_panels"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for letter in "abcdefghij":
        sx, sy, sw, sh = builder.SLOTS_MM[letter]
        if letter == "a":
            left, top, right, bottom = sx-2.5, 0, sx+sw+.5, sy+sh+.5
        else:
            bx0, by0, bx1, by1 = native[letter]["content_bounds_mm"]
            label_y = {"d":69.5, "e":69.5, "f":124.5, "h":124.5}.get(letter, max(sy-1,3.8))
            left = min(sx+bx0, sx-2)-.6
            top = min(sy+sh-by1, label_y-3.3)-.6
            right, bottom = sx+bx1+.6, sy+sh-by0+.6
        width, height = right-left, bottom-top
        root = ET.Element(builder.tag("svg"), {"width":f"{width}mm", "height":f"{height}mm", "viewBox":f"{left*builder.MM} {top*builder.MM} {width*builder.MM} {height*builder.MM}", "version":"1.1"})
        ET.SubElement(root,builder.tag("rect"),{"x":str(left*builder.MM),"y":str(top*builder.MM),"width":str(width*builder.MM),"height":str(height*builder.MM),"fill":"white"})
        if letter == "a":
            builder.maze_image(root,image,bounds)
        else:
            builder.place(root,native[letter],letter,builder.SLOTS_MM[letter])
        stem = folder / f"figure1_panel_{letter}"
        ET.ElementTree(root).write(stem.with_suffix(".svg"),encoding="utf-8",xml_declaration=True)
        jobs.append((letter,stem,width,height))

    def export(job):
        from ca_analysis.browser_export import platform_headless_flags, run_browser
        letter,stem,width,height = job
        html = stem.with_suffix(".html")
        html.write_text("<!doctype html><html><head><meta charset='utf-8'><style>"
            f"@page{{size:{width}mm {height}mm;margin:0}}html,body{{margin:0;width:{width}mm;height:{height}mm;overflow:hidden}}"
            f"body>svg{{display:block;width:{width}mm!important;height:{height}mm!important}}"
            "</style></head><body>"+stem.with_suffix(".svg").read_text(encoding="utf-8")+"</body></html>",encoding="utf-8")
        pdf=stem.with_suffix(".pdf")
        with tempfile.TemporaryDirectory(prefix=f"figure1_panel_{letter}_") as temporary:
            fresh=Path(temporary)/"fresh.pdf"
            run_browser([str(runtime._find_chrome()),*platform_headless_flags(),"--headless=new","--disable-gpu","--no-sandbox","--disable-software-rasterizer","--no-pdf-header-footer","--print-to-pdf-no-header",f"--user-data-dir={Path(temporary)/'browser'}",f"--print-to-pdf={fresh}",html.as_uri()],check=True,capture_output=True,timeout=120)
            runtime._wait_for_file(fresh)
            shutil.copy2(fresh,pdf)
        with fitz.open(pdf) as doc:
            assert len(doc)==1
            assert np.allclose([doc[0].rect.width/builder.MM,doc[0].rect.height/builder.MM],[width,height],atol=.4)
            pix=doc[0].get_pixmap(dpi=600,alpha=False)
            raster=Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
            raster.save(stem.with_suffix(".png"),dpi=(600,600),optimize=True)
            raster.save(stem.with_suffix(".tiff"),dpi=(600,600),compression="tiff_lzw")
        return letter,{ext:str(stem.with_suffix('.'+ext).relative_to(output)) for ext in ["svg","pdf","png","tiff"]}

    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest=dict(pool.map(export,jobs))
    (output/"figure1_individual_panels_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


def rebuild(record: Path, output: Path) -> list[Path]:
    record, output = Path(record).resolve(), Path(output).resolve()
    assert output != record and record not in output.parents, "Outputs must be outside the immutable record"
    inputs = verify_inputs(record)
    output.mkdir(parents=True, exist_ok=True)
    runtime = load("figure1_frozen_rebuilder", record / "source_code/source_runtime.py")
    runtime.VOR_ROOT = record / "source_data"
    runtime.REPO_ROOT = record
    frozen_panels = runtime._install_frozen_panel_modules(record)
    preview = runtime._load_module("record_local_main1_preview", record / "source_code/preview_figure1_assembly.py")
    names = {
        "FIGURE1_REWARD_ACQUISITION_POINTS": "reward_acquisition_day1_month_points_current_plus_new12.csv",
        "FIGURE1_REWARD_ACQUISITION_SUMMARY": "reward_acquisition_day1_month_summary_current_plus_new12.csv",
        "FIGURE1_REWARD_ACQUISITION_CONTRASTS": "reward_acquisition_day1_month_clustered_contrasts_current_plus_new12.csv",
        "FIGURE1_EXPLORATION_POINTS": "exploration_first_chunk_current_plus_new_points.csv",
        "FIGURE1_LONG_BOUT_POINTS": "long_bout_group_points_current_plus_new12.csv",
        "FIGURE1_LONG_BOUT_SUMMARY": "long_bout_group_summary_current_plus_new12.csv",
        "FIGURE1_LONG_BOUT_CONTRASTS": "long_bout_group_clustered_contrasts_current_plus_new12.csv",
        "FIGURE1_ROUTE_SCORE_POINTS": "ed04_session_route_current_plus_new12.csv",
        "FIGURE1_ROUTE_SCORE_SUMMARY": "ed04_route_group_stats_current_plus_new12.csv",
        "FIGURE1_ROUTE_SCORE_CONTRASTS": "ed04_route_stats_current_plus_new12.csv",
    }
    for name, filename in names.items():
        setattr(preview, name, record / "plot_ready" / filename)
    complete = pd.read_csv(record / "plot_ready/figure1_endnode_timelines.csv")

    def extract(nickname, *, lower_threshold=63, upper_threshold=126):
        rows = complete.loc[complete.nickname.eq(nickname) & complete.endnode.between(lower_threshold, upper_threshold)]
        assert len(rows), nickname
        return rows.endnode.to_numpy(int), rows.time_in_maze_s.to_numpy(float)

    frozen_panels.extract_endnode_timeline = extract
    builder = load("figure1_approved_layout", record / "source_code/build_figure1_layout_alternatives.py")
    builder.MAIN = record
    audit = {}
    preview, panels = builder.native_panels(runtime, output, audit)
    panels["b"], timeline_rows = builder.session_timeline(preview)
    ed = load("figure1_frozen_trajectory_panel", record / "source_code/trajectory_panel.py")
    trajectory_data = pd.read_csv(record / "source_data/trajectory_examples.csv.gz")
    trajectory, segment_counts = builder.trajectories(preview, ed, trajectory_data, builder.VARIANTS["A_day1_day91"], "A_day1_day91")
    image = record / "source_data/maze_3d_original.png"
    Image.MAX_IMAGE_PIXELS = 12800 * 8000 + 1
    with Image.open(image) as original:
        bounds = original.getchannel("A").getbbox()
    root = ET.Element(builder.tag("svg"), {"width":"180mm", "height":"196mm", "viewBox":f"0 0 {builder.WIDTH} {builder.HEIGHT}", "version":"1.1"})
    ET.SubElement(root, builder.tag("rect"), {"width":"100%", "height":"100%", "fill":"white"})
    builder.maze_image(root, image, bounds)
    geometry = {}
    for letter, key in builder.MAPPING.items():
        geometry[letter] = builder.place(root, panels[key], letter, builder.SLOTS_MM[letter])
    geometry["b"] = builder.place(root, trajectory, "b", builder.SLOTS_MM["b"])
    stem = output / "figure1_version_record"
    ET.ElementTree(root).write(stem.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)
    builder.export(stem.with_suffix(".svg"), stem)
    with fitz.open(stem.with_suffix(".pdf")) as doc:
        pix = doc[0].get_pixmap(dpi=600, alpha=False)
        raster = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        raster.save(output / "figure1_version_record_600dpi.png", dpi=(600,600), optimize=True)
        raster.save(output / "figure1_version_record_600dpi.tiff", dpi=(600,600), compression="tiff_lzw")
    metadata = {
        "status":"PASS", "schema":"figure1_ten_panel_native_rebuild/v1",
        "source_inputs":inputs,
        "panel_letters":list("abcdefghij"), "geometry":geometry,
        "axes":builder.AXES_MM, "panel_slots":builder.SLOTS_MM,
        "quantitative_display_adjustments":builder.DISPLAY_ADJUSTMENTS,
        "significance_layout":builder.SIGNIFICANCE_AUDIT,
        "drawn_segments":segment_counts, "timeline_labels":timeline_rows,
        "no_master_artwork_read":True, "fixed_illustration":"source_data/maze_3d_original.png",
    }
    (output / "NATIVE_REBUILD.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    native = {letter:panels[key] for letter,key in builder.MAPPING.items()}
    native["b"] = trajectory
    export_individual_panels(builder,runtime,native,image,bounds,output)
    verify_inputs(record)
    return sorted(p for p in output.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = rebuild(args.record, args.output_dir)
    print(json.dumps({"status":"PASS", "outputs":[str(p) for p in paths]}, indent=2))


if __name__ == "__main__":
    main()
