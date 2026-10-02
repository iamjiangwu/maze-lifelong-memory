"""Build Figure 5 from the included level-code and neural-behavior Source Data.

The supported figure build.py launcher supplies the staged inputs and output
directory. No accepted artwork or raw-data processing is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import fitz

RECORD_NAME = "main_figure5_2026-09-25_round10_black_stars"
SVG = "{http://www.w3.org/2000/svg}"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path: Path) -> dict:
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module




def configure(inputs: Path):
    index = json.loads((inputs / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))
    for relative, expected in index["sources"].items():
        actual = info(inputs / relative)
        assert actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"], relative
    engine = load("exchange_frozen_main_runtime", inputs / "source_code/frozen_main_rebuild_runtime.py")
    engine.VOR_ROOT = inputs / "source_data"
    engine.REPO_ROOT = inputs
    engine._install_frozen_panel_modules(inputs)
    publication = engine._load_publication_renderer(inputs, "exchange_frozen_publication")
    module = engine._load_module("exchange_frozen_main5", inputs / "source_code/preview_figure3_assembly.py")
    bundle = inputs / "source_data/figure5_bundle"
    module.FIGURE5_LEVEL_DISTRIBUTION_DIR = bundle
    module.FIGURE5_LEVEL_HEATMAP_PLOT_READY = inputs / "source_data/level_vectors_tree_depth7_corrected36_d1v1.csv"
    module.FIGURE5_LEVEL_HEATMAP_PLOT_READY_SHA256 = sha(module.FIGURE5_LEVEL_HEATMAP_PLOT_READY)
    module.SORTED_LEVEL_EXPLORATION_DIR = bundle
    scope = pd.read_csv(bundle / "figure5_pair_scope_audit.csv")
    rows = pd.read_csv(module.FIGURE5_LEVEL_HEATMAP_PLOT_READY)
    assert len(scope) == 36 and len(rows) == 16901
    validation = {"status": "PASS", "pairs": len(scope), "neurons": len(rows),
                  "source": "record-frozen ordinary Level-code sources"}
    return engine, publication, module, index, validation


def rebuild(inputs: Path, output: Path) -> list[Path]:
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine, publication, module, index, source_validation = configure(inputs)
    order = inputs / "source_data/figure5b_historical_hierarchical_refined_order.npy"
    pv_path = inputs / "source_data/level_pv/level_pv_matrix.csv"
    matrix = pd.read_csv(pv_path).set_index("train_depth").to_numpy(float)
    pairs = pd.read_csv(inputs / "source_data/level_pv/level_pv_pair_matrices.csv")
    rebuilt_matrix = pairs.pivot_table(index="train_depth", columns="test_depth", values="pearson_r", aggfunc="mean").to_numpy()
    assert matrix.shape == (7, 7) and np.allclose(matrix, rebuilt_matrix, rtol=0, atol=1e-15)
    native_renderers = dict(module.PANEL_RENDERERS)
    audit = {"status": "PASS", "source_validation": source_validation,
             "matrix_max_abs_reconstruction_difference": float(np.max(abs(matrix-rebuilt_matrix))),
             "pv_palette": ["#ffffff", "#7976A2", "#4E4A75"], "pv_limits": [0, .65],
             "pv_levels": list(range(7)), "retained_panels": ["a", "b", "f"],
             "retained_native_dimensions": True, "raw_analysis_runs": 0,
             "accepted_artwork_used_as_input": False, "axes": {}}
    audit["pv_pair_support"] = pairs.pivot_table(
        index="train_depth", columns="test_depth", values="pearson_r", aggfunc="count"
    ).to_numpy().tolist()
    audit["pv_aggregation"] = "Cell-wise mean of saved registered-neuron population-vector Pearson correlations across session pairs; each cell has 36 or 32 supported pairs."
    audit["pv_within_neuron_normalization"] = "None; the matrix and its numerical normalization are unchanged from the reviewed ED9 Level-PV sources."
    density = np.load(inputs / "source_data/figure5_bundle/plot_data_cell228_level.npz")
    audit["density_source"] = {"neurons": int(density["obs"].size),
                               "shuffle_histogram_bins": int(density["shuffle_hist"].size)}
    audit["association_statistics"] = json.loads((inputs / "source_data/figure5_bundle/level_map_vs_exploration_efficiency_primary_clean_stats.json").read_text())
    audit["native_typography_pt"] = {"axis_labels": module.AXIS_LABEL_SIZE,
                                     "ticks": module.TICK_LABEL_SIZE,
                                     "titles": module.TITLE_SIZE,
                                     "panel_letters": module.PANEL_ID_SIZE}

    def render_pv(size):
        module._set_common_rc()
        fig = plt.figure(figsize=size)
        # Native geometry; global text size follows the retained Figure 5 panels.
        side = .72 * size[1]
        ax = fig.add_axes([.42/size[0], .22, side/size[0], .72])
        cmap = LinearSegmentedColormap.from_list("ordinary_level_pv", audit["pv_palette"])
        im = ax.pcolormesh(np.arange(8)-.5, np.arange(8)-.5, matrix,
                          cmap=cmap, vmin=0, vmax=.65, shading="flat", rasterized=False)
        for level in range(7):
            ax.add_patch(Rectangle((level-.5, level-.5), 1, 1, fill=False,
                                   edgecolor="#333333", lw=.55))
        ax.set(xlim=(-.5, 6.5), ylim=(6.5, -.5), xticks=range(7), yticks=range(7),
               xlabel="Later level", ylabel="Earlier level")
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Level PV correlation", fontsize=module.TITLE_SIZE, pad=3)
        cax = fig.add_axes([2.42/size[0], .22 + .15/size[1], .07/size[0], (side-.30)/size[1]])
        cb = fig.colorbar(im, cax=cax, ticks=[0, .3, .6])
        cb.solids.set_rasterized(False)
        cb.ax.tick_params(length=1.5, pad=1, labelsize=module.TICK_LABEL_SIZE)
        cb.outline.set_linewidth(.45)
        module._standardize_axes_text(fig)
        fig.canvas.draw()
        title_box = ax.title.get_window_extent(fig.canvas.get_renderer())
        assert title_box.x0 > fig.bbox.x0 and title_box.x1 < fig.bbox.x1
        assert title_box.y1 < fig.bbox.y1
        audit["panel_c_title"] = ax.title.get_text()
        audit["panel_c_title_native_fontsize_pt"] = float(ax.title.get_fontsize())
        audit["panel_c_title_one_line_inside_panel"] = "\n" not in ax.title.get_text()
        return fig

    def render_density(size):
        fig = native_renderers["b"](size)
        # Align the axes to the unchanged exploration plot in the same row.
        fig.subplots_adjust(left=.16, right=.965, bottom=.22, top=.94)
        return fig

    module.PANEL_RENDERERS = {
        "a": native_renderers["a"],
        "b": lambda size: module._render_panel_e(size, refined_order_path=order, refined_order_sha256=sha(order)),
        "c": render_pv,
        "d": render_density,
        "e": native_renderers["f"],
    }
    positions = {"a": (.03, .05, 3.37, 2.58), "b": (3.44, .05, 5.20, 2.58),
                 "c": (.03, 3.05, 2.72, 2.62), "d": (2.86, 3.05, 2.51, 2.62),
                 "e": (5.48, 3.05, 3.16, 2.62)}
    module.PANEL_LAYOUT_SPEC = {letter: dict(x=x, y=y, w=w, h=h, render_w=w, render_h=h, label=letter)
                                for letter, (x, y, w, h) in positions.items()}
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "figure5-ed9-panel-exchange"})
    panel_dir = output / "individual_panels"
    panel_dir.mkdir(exist_ok=True)
    individual_paths = []
    for letter, spec in module.PANEL_LAYOUT_SPEC.items():
        fig = module.PANEL_RENDERERS[letter]((spec["w"], spec["h"]))
        audit["axes"][letter] = [list(map(float, ax.get_position().bounds)) for ax in fig.axes]
        for suffix in ("svg", "pdf", "png"):
            target = panel_dir / f"figure5_panel_{letter}.{suffix}"
            fig.savefig(target, dpi=300, bbox_inches=None, metadata={"Date": None} if suffix=="svg" else None)
            individual_paths.append(target)
        plt.close(fig)
    row_axes = [audit["axes"][letter][0] for letter in "cde"]
    assert np.allclose([bounds[1] for bounds in row_axes], .22, rtol=0, atol=1e-12)
    assert np.allclose([bounds[1]+bounds[3] for bounds in row_axes], .94, rtol=0, atol=1e-12)
    audit["bottom_row_axes_top_and_bottom_aligned"] = True
    pv_svg = ET.parse(panel_dir / "figure5_panel_c.svg").getroot()
    assert not pv_svg.findall(".//" + SVG + "image")
    audit["new_heatmap_and_colorbar_vector_only"] = True
    with matplotlib.rc_context({"savefig.dpi": 600}):
        tree = module.render_assembly_svg(dpi=600)
    # Keep the accepted Figure 5 outer canvas and top margin exactly.
    top_margin = .12
    root = tree.getroot()
    x, y, w, h = map(float, root.attrib["viewBox"].split())
    root.set("viewBox", f"{x:.6f} {y:.6f} {w:.6f} {h+top_margin*72:.6f}")
    height_mm = round(module.EXPORT_WIDTH_MM*(module.CANVAS_SIZE[1]+top_margin)/module.CANVAS_SIZE[0], 3)
    root.set("height", f"{height_mm:.3f}mm")
    for child in root:
        if child.tag in {SVG+"metadata", SVG+"defs"} or (child.tag==SVG+"rect" and child.get("width")=="100%"):
            continue
        child.set("transform", f"translate(0 {top_margin*72:.6f}) {child.get('transform','')}".strip())
    svg = output / "figure5_panel_exchange.svg"
    tree.write(svg, encoding="utf-8", xml_declaration=True)
    png = output / "figure5_panel_exchange_600dpi.png"
    pdf = output / "figure5_panel_exchange.pdf"
    html = publication._export_svg_with_chrome(svg, png, pdf, width_mm=180, height_mm=height_mm, dpi=600)
    tiff = output / "figure5_panel_exchange_600dpi.tiff"
    small = output / "figure5_panel_exchange_300dpi.png"
    preview = output / "figure5_panel_exchange_preview.png"
    with Image.open(png) as image:
        image.save(tiff, compression="tiff_lzw", dpi=(600,600))
        image.resize((image.width//2, image.height//2), Image.Resampling.LANCZOS).save(small, dpi=(300,300))
    with fitz.open(pdf) as document:
        assert len(document)==1
        page = document[0]
        text = page.get_text()
        assert "Level PV correlation" in text and "Exploration efficiency" in text
        page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(preview)
        audit["pdf_page_mm"] = [page.rect.width/72*25.4, page.rect.height/72*25.4]
        audit["pdf_text_selectable"] = len(text)>200
        audit["significance_black"] = all(span["color"]==0 for block in page.get_text("dict")["blocks"]
            if "lines" in block for line in block["lines"] for span in line["spans"] if span["text"].strip()=="***")
        assert audit["significance_black"] and audit["pdf_text_selectable"]
    audit["panel_letters"] = [n.text for n in root if n.tag==SVG+"text"]
    assert audit["panel_letters"]==list("abcde")
    audit["uniform_assembly_panel_scales"] = [node.get("transform") for node in root if node.tag==SVG+"g"]
    assert all("scale(1.00000000)" in item for item in audit["uniform_assembly_panel_scales"])
    geometry = output / "PANEL_GEOMETRY.csv"
    pd.DataFrame([{"panel":k,"x_in":x,"y_in":y+top_margin,"width_in":w,"height_in":h}
                  for k,(x,y,w,h) in positions.items()]).to_csv(geometry,index=False)
    crosswalk = output / "PANEL_CROSSWALK.csv"
    pd.DataFrame([("a","Figure5 a","Level code schematic"), ("b","Figure5 b","Registered-neuron Level maps"),
                  ("c","ED9 review d","Level PV correlation"),
                  ("d","Figure5 c","Neuron correlation density"),
                  ("e","Figure5 f","Level correlation versus exploration efficiency")],
                 columns=["new_panel","source_panel","description"]).to_csv(crosswalk,index=False)
    validation_path = output / "SOURCE_AND_EXPORT_VALIDATION.json"
    write_json(validation_path,audit)
    outputs = [svg,png,pdf,html,tiff,small,preview,geometry,crosswalk,validation_path,*individual_paths]
    write_json(output / "REBUILD_RECEIPT.json", {
        "status":"PASS", "base_record":RECORD_NAME,
        "input_manifest_sha256":sha(inputs/"SOURCE_INPUTS.json"),
        "builder_sha256":sha(Path(__file__).resolve()),
        "output_size_mm":[180,height_mm], "outputs":{p.relative_to(output).as_posix():info(p) for p in outputs},
        "scientific_changes":"None; panels are exchanged without changing estimands, cohorts, numerical values or statistics.",
        "records_and_manuscript_modified":False,
    })
    (output/"README.md").write_text(
        "# Figure 5 / ED9 panel exchange review\n\n"
        "Figure5 a/b remain unchanged; ED9's Level population-vector heatmap becomes c; "
        "the original density histogram becomes d in the combined old d/e space; original f becomes e.\n\n"
        "Rebuild from the self-contained numerical source bundle:\n\n"
        "`python source_code/rebuild_figure5.py --inputs . --output rebuilt`\n\n"
        "Python with NumPy, pandas, SciPy, matplotlib, Pillow, PyMuPDF and Microsoft Edge is required. "
        "Accepted artwork is not an input. Accepted records and manuscripts remain unchanged.\n"
        "The level population-vector matrix is unchanged from the reviewed ED9 source: cell-wise means across 36 or 32 "
        "supported pairs, with no within-neuron normalization, 0-6 level labels, and color limits 0-0.65.\n",
        encoding="utf-8")
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    files=rebuild(args.inputs,args.output)
    print(json.dumps({"status":"PASS","output":str(args.output.resolve()),"artifacts":len(files)}))


if __name__=="__main__":
    main()
