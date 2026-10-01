"""Refine only Figure 1 i/j using the accepted frozen native source builder.

The review bundle contains all declared source inputs and this adapter. No
accepted record, manuscript, or Desktop file is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import fitz
import numpy as np
from matplotlib.container import BarContainer

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = REPO / "docs/figure_version_records/main_figure1_2026-09-25_round13_compact_source"
DEFAULT_OUTPUT = REPO / "output/figures/figure1_ij_spacing_2026-09-25"
NEW_AXES = {"i": [(81.3, 156, 39.7, 31)], "j": [(135.3, 156, 39.7, 31)]}
NEW_SLOTS = {"i": (70, 155, 51, 39), "j": (124, 155, 52, 39)}
BAR_WIDTH_FACTOR = .85
NS = "http://www.w3.org/2000/svg"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze(record, output):
    frozen = output / "frozen_source"
    rows = list(csv.DictReader((record / "SOURCE_INPUTS.csv").open(encoding="utf-8", newline="")))
    for row in rows:
        source = record / row["path"]
        assert sha(source) == row["sha256"], source
        target = frozen / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == row["sha256"], target
        else:
            shutil.copy2(source, target)
    shutil.copy2(record / "SOURCE_INPUTS.csv", frozen / "SOURCE_INPUTS.csv")
    return frozen


def snapshot(ax):
    """Numerical artists excluding the specifically authorized bar widths."""
    bars = [bar for container in ax.containers if isinstance(container, BarContainer) for bar in container.patches]
    return {
        "bar_centers_heights_bottoms_colors": [[b.get_x()+b.get_width()/2, b.get_height(), b.get_y(), list(b.get_facecolor())] for b in bars],
        "collection_offsets": [np.asarray(c.get_offsets()).tolist() for c in ax.collections],
        "collection_segments": [np.asarray(c.get_segments()).tolist() for c in ax.collections if hasattr(c, "get_segments")],
        "line_data": [[np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()] for line in ax.lines],
        "texts": [t.get_text() for t in ax.texts],
        "limits": [list(ax.get_xlim()), list(ax.get_ylim())],
    }


def canonical_group(path, letter):
    root = ET.parse(path).getroot()
    group = next(element for element in root if element.get("id") == f"panel_{letter}")
    body = ET.tostring(group, encoding="unicode")
    ids = list(dict.fromkeys(re.findall(r'\bid="([^"]+)"', body)))
    for index, value in enumerate(sorted(ids, key=len, reverse=True)):
        body = body.replace(value, f"ID{index:04d}")
    # Layout-equivalent saved SVGs can differ only in insignificant whitespace.
    return re.sub(r">\s+<", "><", body)


def build(output, record=None, *, skip_individual=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_source"
    if not frozen.exists():
        assert record is not None, "Initial build requires the accepted source record"
        frozen = freeze(Path(record).resolve(), output)
    entrypoint = load("ij_review_frozen_entrypoint", frozen / "source_code/rebuild_figure1.py")
    entrypoint.verify_inputs(frozen)
    original_load = entrypoint.load
    artist_audit = {}
    geometry_audit = {}

    def adjusted_load(name, path):
        module = original_load(name, path)
        if name == "record_local_main1_preview":
            for letter in ("i", "j"):
                renderer = module.PANEL_RENDERERS[letter]

                def narrower(size, renderer=renderer, letter=letter):
                    fig = renderer(size)
                    ax = fig.axes[0]
                    before = snapshot(ax)
                    bars = [b for container in ax.containers if isinstance(container, BarContainer) for b in container.patches]
                    widths = []
                    for bar in bars:
                        old = bar.get_width()
                        center = bar.get_x() + old / 2
                        new = old * BAR_WIDTH_FACTOR
                        bar.set_width(new)
                        bar.set_x(center-new/2)
                        widths.append({"center": center, "old_width_data": old, "new_width_data": new})
                    after = snapshot(ax)
                    # Floating point center round-off is at most machine precision.
                    for a, b in zip(before["bar_centers_heights_bottoms_colors"], after["bar_centers_heights_bottoms_colors"], strict=True):
                        assert np.allclose(a[:3], b[:3], atol=1e-14, rtol=0) and a[3] == b[3]
                    for key in before:
                        if key != "bar_centers_heights_bottoms_colors":
                            assert before[key] == after[key], (letter, key)
                    artist_audit[letter] = {"authorized_width_changes": widths, "all_other_data_artists_exact": True,
                        "bar_centers_heights_colors_unchanged": True, "before": before, "after": after}
                    return fig

                module.PANEL_RENDERERS[letter] = narrower
        if name == "figure1_approved_layout":
            module.AXES_MM.update(NEW_AXES)
            module.SLOTS_MM.update(NEW_SLOTS)
            original_panel = module.panel_at_final_size

            def measure_panel(preview, fig, letter):
                result = original_panel(preview, fig, letter)
                sx, sy, sw, sh = module.SLOTS_MM[letter]
                renderer = fig.canvas.get_renderer()

                def global_bbox(text):
                    box = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                    return [sx+box.x0*25.4, sy+sh-box.y1*25.4, sx+box.x1*25.4, sy+sh-box.y0*25.4]

                if letter in ("h", "i", "j"):
                    geometry_audit[letter] = {
                        "slot_mm": list(module.SLOTS_MM[letter]),
                        "axes_mm": result["axes_final_mm"],
                        "ylabel_bounds_mm": [global_bbox(ax.yaxis.label) for ax in fig.axes],
                        "xlabel_bounds_mm": [global_bbox(ax.xaxis.label) for ax in fig.axes],
                        "label_font_sizes_pt": [[ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize()] for ax in fig.axes],
                    }
                return result

            module.panel_at_final_size = measure_panel
        return module

    entrypoint.load = adjusted_load
    if skip_individual:
        entrypoint.export_individual_panels = lambda *args: None
    entrypoint.rebuild(frozen, output)
    entrypoint.verify_inputs(frozen)
    for letter in ("i", "j"):
        box = geometry_audit[letter]["ylabel_bounds_mm"][0]
        left = NEW_SLOTS[letter][0]
        assert box[0] >= left, (letter, box, left)
        assert geometry_audit[letter]["label_font_sizes_pt"] == [[6.5, 6.5]]
        geometry_audit[letter]["ylabel_left_clearance_inside_slot_mm"] = box[0] - left
        geometry_audit[letter]["bottom_axis_mm"] = NEW_AXES[letter][0][1]+NEW_AXES[letter][0][3]
        geometry_audit[letter]["bar_physical_width_ratio_to_accepted"] = BAR_WIDTH_FACTOR * 39.7 / 47
    geometry_audit["h_to_ij_axes_clearance_mm"] = 156-max(box[3] for box in geometry_audit["h"]["xlabel_bounds_mm"])
    geometry_audit["i_ylabel_clearance_from_g_right_axis_mm"] = geometry_audit["i"]["ylabel_bounds_mm"][0][0]-67
    geometry_audit["j_ylabel_clearance_from_i_right_axis_mm"] = geometry_audit["j"]["ylabel_bounds_mm"][0][0]-121
    write_json(output / "IJ_ARTIST_PARITY.json", artist_audit)
    write_json(output / "IJ_GEOMETRY_QA.json", geometry_audit)
    unchanged = {}
    if record is not None:
        accepted = Path(record) / "figure1_version_record.svg"
        for letter in "abcdefgh":
            old, new = canonical_group(accepted, letter), canonical_group(output / "figure1_version_record.svg", letter)
            unchanged[letter] = {"svg_native_artists_identical": old == new}
            assert old == new, f"Unrelated panel {letter} changed"
    write_json(output / "UNRELATED_PANEL_PARITY.json", unchanged)
    with fitz.open(output / "figure1_version_record.pdf") as pdf:
        assert len(pdf) == 1
        font_sizes = sorted(set(round(span["size"], 2) for block in pdf[0].get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]))
    review = {"status": "PASS", "scope": "Native widths, bar widths and upward height extension of panels i/j only",
        "accepted_record": str(record) if record else None, "frozen_input_count": len(entrypoint.verify_inputs(frozen)),
        "axes_mm": NEW_AXES, "slots_mm": NEW_SLOTS, "bar_width_factor_data": BAR_WIDTH_FACTOR,
        "bar_physical_width_factor": BAR_WIDTH_FACTOR*39.7/47,
        "unchanged_panels": unchanged, "pdf_font_sizes_pt": font_sizes,
        "data_errorbars_mean_markers_significance_tokens_unchanged": True,
        "export_scale": 1, "promoted": False}
    write_json(output / "REVIEW_VALIDATION.json", review)
    script_copy = output / "source_code/build_figure1_ij_spacing.py"
    script_copy.parent.mkdir(exist_ok=True)
    if Path(__file__).resolve() != script_copy.resolve():
        shutil.copy2(__file__, script_copy)
    (output / "README.md").write_text(
        "# Figure 1 i/j spacing review\n\n"
        "Only panels i and j change. Their native plot widths decrease from 47 to 39.7 mm; "
        "their tops rise from 160 to 156 mm while the shared baseline remains 187 mm, increasing height from 27 to 31 mm. "
        "Bars retain their centers and heights but are 15% narrower in data coordinates. "
        "All point/errorbar artists, statistical comparisons, colors and 6.5 pt typography are preserved. "
        "The y-axis titles fit inside their own slots. Panels a-h retain identical native SVG artists.\n\n"
        "Source-only replay (writes to a new directory):\n\n"
        "```powershell\n"
        "python source_code/build_figure1_ij_spacing.py --frozen-source frozen_source --output-dir <new directory>\n"
        "```\n\n"
        "The frozen source is copied verbatim from the accepted Round 13 declared inputs; "
        "no composite artwork is used to draw the figure. Accepted artwork is read only for the unrelated-panel QA comparison. "
        "REVIEW_VALIDATION.json, IJ_ARTIST_PARITY.json and IJ_GEOMETRY_QA.json contain the checks. "
        "This is a review export, with no figure-record, manuscript or Desktop changes.\n",
        encoding="utf-8",
    )
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-individual", action="store_true")
    args = parser.parse_args()
    record = args.record
    if args.frozen_source:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir.resolve() / "frozen_source"
        if destination != args.frozen_source.resolve():
            assert not destination.exists()
            shutil.copytree(args.frozen_source, destination)
        record = None
    print(json.dumps(build(args.output_dir, record, skip_individual=args.skip_individual), indent=2))


if __name__ == "__main__":
    main()
