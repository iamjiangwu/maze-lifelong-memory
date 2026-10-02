"""ED01 Round 24 display transform on a freshly built native source figure.

Receives the frozen Round 22 renderer and its record-local numerical tables.
No accepted figure, review image, PDF, SVG or archived record is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image

LABEL_MAP = {old: chr(ord(old) - 1) for old in "bcdefghijk"}


def artist_signature(axis):
    digest = hashlib.sha256()
    for value in (axis.get_xlim(), axis.get_ylim(), axis.get_xscale(), axis.get_yscale()):
        digest.update(repr(value).encode())
    for line in axis.lines:
        for values in (line.get_xdata(), line.get_ydata()):
            digest.update(np.asarray(values).tobytes())
        digest.update(repr((line.get_color(), line.get_linewidth(), line.get_linestyle(), line.get_marker())).encode())
    for collection in axis.collections:
        for values in (collection.get_offsets(), collection.get_facecolors(), collection.get_edgecolors(), collection.get_linewidths()):
            digest.update(np.asarray(values).tobytes())
        for path in collection.get_paths():
            digest.update(path.vertices.tobytes())
    for patch in axis.patches:
        digest.update(patch.get_path().vertices.tobytes())
        digest.update(repr((patch.get_facecolor(), patch.get_edgecolor(), patch.get_linewidth())).encode())
    for text in axis.findobj(plt.Text):
        position = tuple(round(float(value), 8) for value in text.get_position())
        digest.update(repr((text.get_text(), text.get_fontsize(), text.get_color(), text.get_rotation(), position)).encode())
    return digest.hexdigest()


def render(source_renderer, data, output: Path):
    capture = {}
    original = source_renderer.add_panel_letters

    def remember(fig, groups):
        capture["groups"] = groups
        return original(fig, groups)

    source_renderer.add_panel_letters = remember
    try:
        fig, boxes, derived = source_renderer.build_figure(data)
    finally:
        source_renderer.add_panel_letters = original
    groups = capture["groups"]
    assert list(groups) == list("abcdefghijk")
    width, old_height = map(float, fig.get_size_inches())
    removed = set(groups["a"])
    assert len(removed) == 5
    retained = [axis for axis in fig.axes if axis not in removed]
    fig.canvas.draw()
    old_positions = {axis: axis.get_position().frozen() for axis in retained}
    signatures = {axis: artist_signature(axis) for axis in retained}
    letters = {text.get_text(): text for text in fig.texts
               if text.get_text() in "abcdefghijk" and len(text.get_text()) == 1}
    assert set(letters) == set("abcdefghijk")
    for axis in removed:
        axis.remove()
    letters["a"].remove()
    for old, new in LABEL_MAP.items():
        letters[old].set_text(new)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [axis.get_tightbbox(renderer) for axis in retained]
    visible.extend(text.get_window_extent(renderer) for text in fig.texts if text.get_visible())
    new_height = max(box.y1 for box in visible if box is not None) / fig.dpi + 4 / 72
    # An axis-off legend carrier has a wide, invisible axes rectangle. Include
    # its visible legend, not that empty rectangle, in horizontal page bounds.
    horizontal_visible = [axis.get_tightbbox(renderer) for axis in retained if axis.axison]
    horizontal_visible.extend(axis.get_legend().get_window_extent(renderer)
                              for axis in retained if not axis.axison and axis.get_legend() is not None)
    horizontal_visible.extend(text.get_window_extent(renderer) for text in fig.texts if text.get_visible())
    rightmost_artist_inches = max(box.x1 for box in horizontal_visible if box is not None) / fig.dpi
    new_width = rightmost_artist_inches + 4 / 72
    assert 5 < new_height < old_height - 1
    assert 6.5 < new_width < width
    factor = old_height / new_height
    xfactor = width / new_width
    text_positions = {text: text.get_position() for text in fig.texts}
    # Keep the original logical x transform bit-for-bit intact. The native
    # savefig viewport below narrows the output page, without a second floating
    # point rescaling of every x coordinate or any finished-artwork operation.
    fig.set_size_inches(width, new_height, forward=True)
    for axis, box in old_positions.items():
        axis.set_position((box.x0, box.y0 * factor, box.width, box.height * factor))
    for text, (x, y) in text_positions.items():
        text.set_position((x, y * factor))
    fig.canvas.draw()
    geometry = []
    for old, new in LABEL_MAP.items():
        for index, axis in enumerate(groups[old]):
            before = np.array(old_positions[axis].bounds) * (width, old_height, width, old_height)
            after = np.array(axis.get_position().bounds) * (width, new_height, width, new_height)
            assert np.allclose(before, after, rtol=0, atol=1e-10)
            assert signatures[axis] == artist_signature(axis)
            geometry.append({"old_panel": old, "new_panel": new, "axis_index": index,
                             "x_inches": after[0], "y_from_bottom_inches": after[1],
                             "width_inches": after[2], "height_inches": after[3],
                             "plot_artists_unchanged": True})
    output.mkdir(parents=True, exist_ok=True)
    final_renderer = fig.canvas.get_renderer()
    final_visible = [axis.get_tightbbox(final_renderer) for axis in retained if axis.axison]
    final_visible.extend(axis.get_legend().get_window_extent(final_renderer)
                         for axis in retained if not axis.axison and axis.get_legend() is not None)
    final_visible.extend(text.get_window_extent(final_renderer) for text in fig.texts if text.get_visible())
    final_rightmost_inches = max(box.x1 for box in final_visible if box is not None) / fig.dpi
    right_margin_points = (new_width - final_rightmost_inches) * 72
    assert abs(right_margin_points - 4) < 1e-8
    stem = output / "extended_data_figure1_source_rebuild_round24"
    with plt.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(stem.with_suffix("." + suffix), dpi=600,
                        bbox_inches=Bbox.from_bounds(0, 0, new_width, new_height), pad_inches=0)
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(stem.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    pd.DataFrame(geometry).to_csv(output / "RETAINED_AXES_GEOMETRY.csv", index=False)
    layout = []
    for old, new in LABEL_MAP.items():
        x0, y0, x1, y1 = boxes[old]
        layout.append({"panel": new, "old_panel": old, "x0_fraction": x0 * xfactor,
                       "y0_fraction": y0 * factor, "x1_fraction": x1 * xfactor,
                       "y1_fraction": y1 * factor, "width_fraction": (x1 - x0) * xfactor,
                       "height_fraction": (y1 - y0) * factor,
                       "placement": "native matplotlib axes; no panel image transform",
                       "nonuniform_scale": False})
    pd.DataFrame(layout).to_csv(output / "LAYOUT_AUDIT.csv", index=False)
    source_audit = []
    source_labels = {"b-c": "a-b", "d-e": "c-d", "f-rows": "e-rows", "f-stats": "e-stats",
                     "g-rows": "f-rows", "g-stats": "f-stats", "h-rows": "g-rows", "h-reference": "g-reference",
                     "i-selection": "h-selection", "i-naive": "h-naive", "i-return": "h-return", "j": "i", "k": "j"}
    for old, (path, stage) in source_renderer.SOURCE_FILES.items():
        source_audit.append({"panel": source_labels.get(old, "retired-a"), "old_panel": old,
                             "source_path": Path(path).relative_to(Path(path).parents[1]).as_posix(),
                             "source_stage": stage, "displayed": old != "a",
                             "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                             "panel_svg_reused": False})
    pd.DataFrame(source_audit).to_csv(output / "SOURCE_DATA_AUDIT.csv", index=False)
    pd.DataFrame([{"old_panel": old, "new_panel": new} for old, new in LABEL_MAP.items()]).to_csv(output / "PANEL_LABEL_MAP.csv", index=False)
    audit = derived["axis_scale_audit"].loc[lambda frame: frame["panel"].ne("a")].copy()
    audit.insert(0, "old_panel", audit["panel"])
    audit["panel"] = audit["panel"].map(LABEL_MAP)
    audit.to_csv(output / "AXIS_SCALE_AUDIT.csv", index=False)
    for key in ("accumulation_summary", "speed_summary", "time_efficiency_summary", "tile_efficiency_summary"):
        derived[key].to_csv(output / f"{key}.csv", index=False)
    with fitz.open(stem.with_suffix(".pdf")) as doc:
        assert len(doc) == 1 and not doc[0].get_images(full=True)
        spans = [s for block in doc[0].get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for s in line["spans"]]
        observed = [s["text"] for s in spans if len(s["text"]) == 1
                    and s["text"] in "abcdefghijk" and abs(s["size"] - 11.5) < 0.01]
        assert observed == list("abcdefghij"), observed
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(output / "ed01_source_preview.png")
    svg = ET.parse(stem.with_suffix(".svg")).getroot()
    assert not any(node.tag.endswith("image") for node in svg.iter())
    result = {"status": "PASS", "panel_letters": list("abcdefghij"), "panel_label_map": LABEL_MAP,
              "canvas_inches": [new_width, new_height], "retained_axes": len(geometry),
              "parent_canvas_inches": [width, new_height],
              "right_margin_points": right_margin_points,
              "rightmost_artist_inches": final_rightmost_inches,
              "removed_blank_width_inches": width - new_width,
              "canvas_trim_only_no_finished_artwork_crop": True,
              "export_policy": "native source viewport; original logical x transforms unchanged",
              "retained_artists_and_physical_geometry_unchanged": True,
              "scientific_source": "record-local numerical tables and frozen native renderer",
              "retired_panel_a": "Empty geometry placeholders are removed before export; no retired trajectory inputs are loaded."}
    (output / "RENDER_VALIDATION_ROUND24.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    return result
