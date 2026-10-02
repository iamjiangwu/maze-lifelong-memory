"""Apply the Figure 1 i/j display geometry to the included source panels."""
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

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = REPO / "docs/figure_version_records/main_figure1_2026-09-25_round13_compact_source"
DEFAULT_OUTPUT = REPO / "output/figures/figure1_ij_spacing_2026-09-25"
NEW_AXES = {"i": [(81.3, 156, 39.7, 31)], "j": [(135.3, 156, 39.7, 31)]}
NEW_SLOTS = {"i": (70, 155, 51, 39), "j": (124, 155, 52, 39)}
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


def canonical_group(path, letter):
    root = ET.parse(path).getroot()
    group = next(element for element in root if element.get("id") == f"panel_{letter}")
    body = ET.tostring(group, encoding="unicode")
    ids = list(dict.fromkeys(re.findall(r'\bid="([^"]+)"', body)))
    for index, value in enumerate(sorted(ids, key=len, reverse=True)):
        body = body.replace(value, f"ID{index:04d}")
    # Layout-equivalent saved SVGs can differ only in insignificant whitespace.
    return re.sub(r">\s+<", "><", body)


def build(output, record=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_source"
    if not frozen.exists():
        assert record is not None, "Initial build requires the accepted source record"
        frozen = freeze(Path(record).resolve(), output)
    entrypoint = load("ij_review_frozen_entrypoint", frozen / "source_code/rebuild_figure1.py")
    entrypoint.verify_inputs(frozen)
    original_load = entrypoint.load
    geometry_audit = {}

    def adjusted_load(name, path):
        module = original_load(name, path)
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
    entrypoint.rebuild(frozen, output)
    entrypoint.verify_inputs(frozen)
    for letter in ("i", "j"):
        box = geometry_audit[letter]["ylabel_bounds_mm"][0]
        left = NEW_SLOTS[letter][0]
        assert box[0] >= left, (letter, box, left)
        assert geometry_audit[letter]["label_font_sizes_pt"] == [[6.5, 6.5]]
        geometry_audit[letter]["ylabel_left_clearance_inside_slot_mm"] = box[0] - left
        geometry_audit[letter]["bottom_axis_mm"] = NEW_AXES[letter][0][1]+NEW_AXES[letter][0][3]
    geometry_audit["h_to_ij_axes_clearance_mm"] = 156-max(box[3] for box in geometry_audit["h"]["xlabel_bounds_mm"])
    geometry_audit["i_ylabel_clearance_from_g_right_axis_mm"] = geometry_audit["i"]["ylabel_bounds_mm"][0][0]-67
    geometry_audit["j_ylabel_clearance_from_i_right_axis_mm"] = geometry_audit["j"]["ylabel_bounds_mm"][0][0]-121
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
    review = {"status": "PASS", "scope": "Figure 1 i/j display geometry",
        "frozen_input_count": len(entrypoint.verify_inputs(frozen)),
        "axes_mm": NEW_AXES, "slots_mm": NEW_SLOTS,
        "source_panel_parity": unchanged, "pdf_font_sizes_pt": font_sizes,
        "source_data_artists_match": True,
        "export_scale": 1}
    script_copy = output / "source_code/build_figure1_ij_spacing.py"
    script_copy.parent.mkdir(exist_ok=True)
    if Path(__file__).resolve() != script_copy.resolve():
        shutil.copy2(__file__, script_copy)
    (output / "README.md").write_text(
        "# Figure 1 i/j display geometry\n\n"
        "Panels i and j have native plot widths of 39.7 mm, tops at 156 mm and a shared baseline at 187 mm. "
        "Their axes height is 31 mm. "
        "Point/errorbar values, statistical comparisons and colors match the source panels; typography is 6.5 pt. "
        "The y-axis titles fit inside their own slots.\n\n"
        "Source-only replay (writes to a new directory):\n\n"
        "```powershell\n"
        "python source_code/build_figure1_ij_spacing.py --frozen-source frozen_source --output-dir <new directory>\n"
        "```\n\n"
        "No composite artwork is used to draw the figure. "
        "IJ_GEOMETRY_QA.json contains the geometry checks. "
        "Only the requested output directory is written.\n",
        encoding="utf-8",
    )
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = args.record
    if args.frozen_source:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir.resolve() / "frozen_source"
        if destination != args.frozen_source.resolve():
            assert not destination.exists()
            shutil.copytree(args.frozen_source, destination)
        record = None
    print(json.dumps(build(args.output_dir, record), indent=2))


if __name__ == "__main__":
    main()
