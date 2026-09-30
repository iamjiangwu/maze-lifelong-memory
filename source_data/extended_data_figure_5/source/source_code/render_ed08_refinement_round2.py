#!/usr/bin/env python3
"""Native append-only ED8 candidate with the full panel-c metric label."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ED8 = (
    REPO
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed08_reward_proximal_controls"
)
PARENT_RENDERER = ED8 / "revision_round10_2026-08-11" / "render_ed08_round10.py"
PARENT_MANIFEST = ED8 / "revision_round10_2026-08-11" / "ED08_ROUND10_RENDER_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def render(output_dir: Path) -> dict[str, Any]:
    parent = load_module("ed08_round2_parent_round10", PARENT_RENDERER)
    r = parent.r
    output_dir = output_dir.resolve()
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    root_stem = output_dir / "extended_data_figure8_refinement_round2"
    r.OUTPUT = output_dir
    r.PANEL_DIR = panel_dir
    r.ROOT_STEM = root_stem
    r.MANIFEST = output_dir / "ED08_REFINEMENT_ROUND2_MANIFEST.json"

    inherited_drawers = dict(r.DRAWERS)

    def draw_c(fig, data, base):
        axes = inherited_drawers["c"](fig, data, base)
        axes[0].set_ylabel("Population vector correlation", labelpad=1.5)
        return axes

    r.DRAWERS = dict(inherited_drawers)
    r.DRAWERS["c"] = draw_c
    r.configure_style()
    base = r.load_base_renderer()
    data = r.read_data(base)
    panel_a, panel_a_geometry = parent.r9.r8.render_panel_a_bundle(data, base)
    bundles, geometry = r.render_quantitative_panels(data, base)
    geometry = panel_a_geometry + geometry
    bundles["a"] = panel_a
    bundles = {label: bundles[label] for label in "abcdefghi"}
    root_pdf = r.compose_root_pdf(bundles)
    root_svg = r.compose_root_svg(bundles)
    root_png = root_stem.with_suffix(".png")
    root_tiff = root_stem.with_suffix(".tiff")
    r.render_pdf_to_rasters(root_pdf, root_png, root_tiff)
    # The inherited pipeline keeps a 300-dpi convenience PNG. Candidate QA and
    # promotion use a full 600-dpi root PNG, matching the TIFF companion.
    with fitz.open(root_pdf) as document:
        document[0].get_pixmap(dpi=600, alpha=False).save(root_png)

    parent_payload = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "schema": "ed08-native-panel-c-full-label-refinement-round2/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "parent_renderer": record(PARENT_RENDERER),
        "parent_manifest": record(PARENT_MANIFEST),
        "scientific_source_authorities": parent_payload.get("source_authorities", {}),
        "display_to_source_role": parent_payload.get("display_to_source_role"),
        "layout": parent_payload.get("layout"),
        "typography_points": parent_payload.get("typography_points"),
        "axes_geometry_parent": parent_payload.get("axes_geometry"),
        "axes_geometry_candidate": geometry,
        "display_change": {
            "panel": "c",
            "old": "PV correlation",
            "new": "Population vector correlation",
            "artist_level": True,
            "quantitative_recomputation": False,
            "layout_changed": False,
        },
        "panels": {
            label: [record(path) for path in paths] for label, paths in bundles.items()
        },
        "root_outputs": [record(path) for path in (root_svg, root_pdf, root_png, root_tiff)],
    }
    r.MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = render(HERE / "ed08")
    print(json.dumps({"status": result["status"], "root_outputs": result["root_outputs"]}, indent=2))
