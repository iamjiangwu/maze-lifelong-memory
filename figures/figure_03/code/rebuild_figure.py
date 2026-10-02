"""Rebuild Figure 3 from the included source bundle using the shared engine."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(bundle, output):
    for row in json.loads((bundle / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))["files"]:
        assert sha(bundle / row["path"]) == row["sha256"], row["path"]
    driver = load(bundle / "source_code/rebuild_current_main_figures.py", "figure3_cycle_driver")
    driver.REPO_ROOT = bundle
    driver._render_main3(bundle / "frozen_source", output, {})
    import fitz
    for suffix in (".svg", ".pdf"):
        shutil.copy2(output / ("main_figure_3_source_rebuild"+suffix), output / ("figure3_cycle_label"+suffix))
    for suffix in (".png", ".tiff"):
        shutil.copy2(output / ("main_figure_3_source_rebuild_600dpi"+suffix), output / ("figure3_cycle_label_600dpi"+suffix))
    with fitz.open(output / "figure3_cycle_label.pdf") as document:
        document[0].get_pixmap(dpi=300, alpha=False).save(output / "figure3_cycle_label_300dpi.png")
