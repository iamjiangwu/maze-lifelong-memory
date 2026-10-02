"""Build Extended Data Figure 10 from the included source bundle.

The supported build.py launcher verifies and stages inputs, then calls
build(bundle, output). Raw-workspace preparation is upstream of this release.
"""

from __future__ import annotations

import hashlib

import importlib.util

import json

import shutil

import sys

import fitz

import pandas as pd

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def build(bundle, output):
    inputs = json.loads((bundle / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))
    for row in inputs["files"]:
        assert sha(bundle / row["path"]) == row["sha256"], row["path"]
    adapter = bundle / "source_code/rebuild_current_extended_data_figures.py"
    spec = importlib.util.spec_from_file_location("ed10_absolute_adapter", adapter)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    driver.REPO_ROOT = bundle
    # All figures share the same plotting runtime.
    shared = bundle.parents[1] / "shared"
    driver.VOR_ROOT = shared
    output.mkdir(parents=True, exist_ok=True)
    frozen = bundle / "frozen_source/extended_data_fig10_round18_inputs"
    with driver.frozen_stage(frozen) as stage:
        result = driver.render_ed10(frozen, output, stage)
    assert result["status"] == "PASS"
    stem = output / "assembly/round18/extended_data_figure10_source_rerender_round18"
    for suffix in ("svg", "pdf"):
        shutil.copy2(stem.with_suffix("." + suffix), output / ("ed10_absolute_heatmap." + suffix))
    for suffix in ("png", "tiff"):
        shutil.copy2(stem.with_suffix("." + suffix), output / ("ed10_absolute_heatmap_600dpi." + suffix))
    with fitz.open(stem.with_suffix(".pdf")) as document:
        document[0].get_pixmap(dpi=300, alpha=False).save(output / "ed10_absolute_heatmap_300dpi.png")
    display = pd.read_csv(output / "source_data/ED10_PANEL_A_ABSOLUTE_DISPLAY.csv", keep_default_na=False)
    assert len(display) == 18
    assert (display.absolute_pearson_r == display.pooled_pearson_r.abs()).all()
    assert display.stars.ne("").sum() == 12
    assert display.absolute_pearson_r.between(0, .8).all()
    write_json(output / "DISPLAY_VALIDATION.json", {
        "status": "PASS", "cell_count": 18, "all_values_exact_absolute_correlations": True,
        "star_count": 12, "scale": [0, .8], "label": "Absolute Pearson correlation",
        "renderer_audit_status": result["status"], "all_frozen_input_hashes_verified": True,
    })
    print(json.dumps({"status": "PASS", "output": str(output)}))
