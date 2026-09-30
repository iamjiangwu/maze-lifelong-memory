#!/usr/bin/env python3
"""Native append-only ED9 candidate for placement, tick, and label refinements."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ED9 = (
    REPO
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed09_tree_depth_level_controls"
)
CORRECTION = ED9 / "correction_2026-08-09"
ROUND6_RENDERER = CORRECTION / "render_ed09_narrow_typography_round6.py"
ROUND12_RENDERER = CORRECTION / "render_ed09_significance_round12.py"
ROUND20_MANIFEST = CORRECTION / "open_terms_levels_round20_2026-08-10" / "RENDER_MANIFEST.json"
PARENT_RECORD = REPO / "docs/figure_version_records/extended_data_fig09_2026-08-11_round22_complete_metadata_closure"
DPI = 600


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


def copy_parent_panels(panel_dir: Path) -> None:
    source = PARENT_RECORD / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in {".svg", ".pdf", ".png", ".tiff"}:
            shutil.copy2(path, panel_dir / path.name)


def configure_round6_style(round6: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": round6.FONT["base"],
            "axes.labelsize": round6.FONT["axis_label"],
            "xtick.labelsize": round6.FONT["tick"],
            "ytick.labelsize": round6.FONT["tick"],
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def native_context(round6: Any, panel_dir: Path):
    round3 = round6.load_module("ed09_round2_round3", round6.ROUND3_RENDERER)
    round4 = round6.load_module("ed09_round2_round4", round6.ROUND4_RENDERER)
    round3.ROUND = HERE
    round3.PANELS = panel_dir
    round3.SOURCE = round6.SOURCE
    base = round3.load_base_renderer()
    base.SOURCE = round6.SOURCE
    old = base.load_old_renderer()
    old.ROOT = round6.SOURCE
    old.RENDER_ROOT = panel_dir
    return round3, round4, old


def render_panel_f(round6: Any, panel_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    round12 = load_module("ed09_round2_round12", ROUND12_RENDERER)
    old = load_module("ed09_round2_old_for_f", round12.OLD_RENDERER)
    old.ROOT = round12.ROUND4_SOURCE
    old.RENDER_ROOT = panel_dir
    original_save = old.save_figure

    def save_with_tick_offset(fig, stem, description):
        axis = fig.axes[0]
        labels = [label for label in axis.get_xticklabels() if label.get_text().startswith("Vertical")]
        if len(labels) != 1:
            raise AssertionError(f"Expected one Vertical tick artist, found {len(labels)}")
        source = labels[0]
        replacement = axis.text(
            *source.get_position(),
            source.get_text(),
            transform=source.get_transform()
            + ScaledTranslation(-3.0 / 72.0, 0.0, fig.dpi_scale_trans),
            ha=source.get_horizontalalignment(),
            va=source.get_verticalalignment(),
            multialignment=source._get_multialignment(),
            fontsize=source.get_fontsize(),
            fontfamily=source.get_fontfamily(),
            color=source.get_color(),
            clip_on=False,
        )
        replacement.set_gid("vertical_tick_offset_minus_3pt")
        source.set_visible(False)
        return original_save(fig, stem, description)

    old.save_figure = save_with_tick_offset
    try:
        return round12.render_panel_f(old, round6)
    finally:
        old.save_figure = original_save


def render_panel_l(round6: Any, panel_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    round3, round4, old = native_context(round6, panel_dir)
    width_px, height_px = round6.PANEL_SIZE_PX["J1"]
    fig, audit = round4.draw_j1(
        round3,
        old,
        {"w_in": width_px / DPI, "h_in": height_px / DPI},
    )
    axis = fig.axes[0]
    zero_lines = [
        line
        for line in list(axis.lines)
        if len(line.get_ydata()) == 2
        and all(abs(float(value)) < 1e-12 for value in line.get_ydata())
        and str(line.get_color()).lower() == "#bfbfbf"
    ]
    if len(zero_lines) != 1:
        raise AssertionError(f"Expected one inherited grey zero line, found {len(zero_lines)}")
    zero_lines[0].remove()
    axis.set_xlabel("No. of experiences")
    axis.set_ylabel("Level map correlation")
    round6.normalize_panel(fig, "J1")
    paths = old.save_figure(
        fig,
        "ED9_J1_long_gap_experience",
        "ED9 panel l with open metric wording and No. of experiences",
    )
    audit.update(
        {
            "x_axis_label": "No. of experiences",
            "grey_zero_line_removed": True,
            "artist_level": True,
        }
    )
    return paths, audit


def render(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    panel_dir = output_dir / "individual_panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_parent_panels(panel_dir)
    round6 = load_module("ed09_round2_round6", ROUND6_RENDERER)
    configure_round6_style(round6)
    panel_f_paths, panel_f_audit = render_panel_f(round6, panel_dir)
    panel_l_paths, panel_l_audit = render_panel_l(round6, panel_dir)

    parent = json.loads(ROUND20_MANIFEST.read_text(encoding="utf-8"))
    panel_rows = copy.deepcopy(parent["panels"])
    for row in panel_rows:
        for suffix in ("svg", "pdf", "png", "tiff"):
            source_name = Path(str(row[suffix])).name
            target = panel_dir / source_name
            if not target.is_file():
                raise FileNotFoundError(target)
            row[suffix] = target.relative_to(REPO).as_posix()
            row[f"{suffix}_sha256"] = sha256(target)
            row[f"{suffix}_bytes"] = target.stat().st_size

    placements = copy.deepcopy(parent["placements"])
    c1 = [row for row in placements if row["role"] == "C1"]
    if len(c1) != 1 or int(c1[0]["placed_px"]["x"]) != 125:
        raise AssertionError("Unexpected parent C1 placement")
    c1[0]["placed_px"]["x"] = 75

    assembly = load_module("ed09_round2_assembly", ROUND12_RENDERER)
    assembly.HERE = REPO
    assembly.CANVAS_PX = (3900, 4010)
    stem = output_dir / "extended_data_figure9_refinement_round2"
    assembly.MASTER_SVG = stem.with_suffix(".svg")
    assembly.MASTER_PDF = stem.with_suffix(".pdf")
    assembly.MASTER_PNG = stem.with_suffix(".png")
    assembly.MASTER_TIFF = stem.with_suffix(".tiff")
    assembly.MASTER_PREVIEW = output_dir / "extended_data_figure9_refinement_round2_300dpi.png"
    assembly.assemble_raster(panel_rows, placements)
    assembly.compose_master_svg(panel_rows, placements)
    assembly.compose_master_pdf(panel_rows, placements)

    manifest = {
        "schema": "ed09-native-refinement-round2/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "parent_manifest": record(ROUND20_MANIFEST),
        "parent_record": PARENT_RECORD.relative_to(REPO).as_posix(),
        "display_changes": {
            "C1": {"placement_x_parent_px": 125, "placement_x_candidate_px": 75},
            "B2": {"vertical_tick_offset_pt": -3.0, "artist_level": True},
            "J1": {"x_axis_label": "No. of experiences", "artist_level": True},
        },
        "scientific_recomputation": False,
        "panel_f_audit": panel_f_audit,
        "panel_l_audit": panel_l_audit,
        "panel_f_outputs": {key: record(path) for key, path in panel_f_paths.items()},
        "panel_l_outputs": {key: record(path) for key, path in panel_l_paths.items()},
        "panels": panel_rows,
        "placements": placements,
        "root_outputs": [
            record(path)
            for path in (
                assembly.MASTER_SVG,
                assembly.MASTER_PDF,
                assembly.MASTER_PNG,
                assembly.MASTER_TIFF,
            )
        ],
    }
    (output_dir / "ED09_REFINEMENT_ROUND2_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = render(HERE / "ed09")
    print(json.dumps({"status": result["status"], "root_outputs": result["root_outputs"]}, indent=2))
