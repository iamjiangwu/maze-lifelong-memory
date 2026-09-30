"""Validate the single Day 1 / Day 91 Figure 1 draft at final size."""
from pathlib import Path
import base64
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures/figure1_day1_day91_compact_2026-09-25_refined"
NS = "{http://www.w3.org/2000/svg}"
MM = 72/25.4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(text):
    return np.asarray([float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)])


def main():
    manifest = json.loads((OUT/"manifest.json").read_text())
    hashes = json.loads((OUT/"OUTPUT_HASHES.json").read_text())
    checks = {}
    checks["sources_unchanged"] = all(sha(p)==h for p,h in manifest["source_hashes"].items())
    checks["outputs_match_hashes"] = all(sha(OUT/p)==h for p,h in hashes.items())
    checks["builder_matches"] = sha(manifest["builder"]["path"])==manifest["builder"]["sha256"]
    checks["only_day1_day91"] = len(manifest["variants"])==1 and next(iter(manifest["variants"].values()))["sessions"]==["434RRday1","434-3day1"]
    adjustments = manifest["quantitative_display_adjustments"]
    checks["native_data_bars_errorbars_annotations_preserved"] = all(v["data_and_annotation_hash_before"]==v["data_and_annotation_hash_after"] for v in adjustments.values())
    source_parity = json.loads((OUT/"TIMELINE_SOURCE_PARITY.json").read_text())
    checks["all_complete_rewarded_events_match_accepted_artwork"] = source_parity["status"]=="PASS" and sum(s["complete_event_count"] for s in source_parity["sessions"])==4380
    stem = OUT/"figure1_A_day1_day91"
    tree = ET.parse(stem.with_suffix(".svg")).getroot()
    groups = {n.get("id"):n for n in tree if n.get("id", "").startswith("panel_")}
    checks["ten_panel_groups"] = set(groups)=={"panel_"+c for c in "abcdefghij"}
    checks["native_scale_one"] = all(g.get("transform", "").endswith("scale(1.0)") for k,g in groups.items() if k!="panel_a")
    ids = [n.get("id") for n in tree.iter() if n.get("id")]
    checks["unique_svg_ids"] = len(ids)==len(set(ids))
    images = list(tree.iter(NS+"image"))
    checks["one_original_raster"] = len(images)==1
    raw = base64.b64decode(images[0].get("{http://www.w3.org/1999/xlink}href").split(",",1)[1])
    checks["supplied_png_bytes_unchanged"] = hashlib.sha256(raw).hexdigest()==sha(OUT/"inputs/maze_3d_original.png")
    mapping = {"e":"c", "f":"d", "g":"g", "h":"h", "i":"i", "j":"j"}
    axes = {new:adjustments[old]["axes_final_mm"] for new,old in mapping.items()}
    for letter,expected in axes.items():
        group = groups["panel_"+letter]
        offset = numbers(group.get("transform"))[:2]/MM
        actual = []
        for cp in group.iter(NS+"clipPath"):
            for rect in cp.iter(NS+"rect"):
                actual.append([float(rect.get("x"))/MM+offset[0],float(rect.get("y"))/MM+offset[1],float(rect.get("width"))/MM,float(rect.get("height"))/MM])
        checks["actual_svg_axis_geometry_"+letter] = all(any(np.allclose(a,list(e.values()),atol=1e-5) for a in actual) for e in expected)
    checks["six_scatter_frames_47_by_19mm"] = all(np.allclose([a["width_mm"],a["height_mm"]],[47,19]) for p in ["e","h"] for a in axes[p])
    checks["first_bout_and_latent_share_row"] = abs(axes["f"][0]["y_mm"]-axes["h"][0]["y_mm"])<1e-7
    checks["last_three_axes_top_and_bottom_aligned"] = all(np.allclose([axes[p][0]["y_mm"],axes[p][0]["height_mm"]],[160,27]) for p in ["g","i","j"])
    checks["left_chart_edges_aligned"] = all(abs(axes["f"][0][k]-axes["g"][0][k])<1e-7 for k in ["x_mm","width_mm"])
    checks["right_columns_aligned"] = all(abs(axes[p][idx]["x_mm"]-x)<1e-7 for p,idx,x in [("e",0,74),("e",1,128),("e",2,74),("e",3,128),("h",0,74),("h",1,128),("i",0,74),("j",0,128)])
    tick_shapes = []
    for node in groups["panel_c"].iter(NS+"g"):
        if "session_tick_" in node.get("id", ""):
            xy = numbers(next(node.iter(NS+"path")).get("d")).reshape(-1,2)
            tick_shapes.append(np.ptp(xy,axis=0)/MM)
    checks["ten_legible_timeline_ticks"] = len(tick_shapes)==10 and np.allclose(tick_shapes,[.7,3],atol=1e-5)
    wall_nodes = [n for n in tree.iter(NS+"g") if "maze_wall_" in n.get("id", "")]
    wall_styles = [next(n.iter(NS+"path")).get("style", "") for n in wall_nodes]
    checks["three_maze_wall_weights_match"] = len(wall_styles)==3 and all("stroke-width: 0.62" in s for s in wall_styles)
    drop_paths = [n for n in tree.iter(NS+"path") if "water_outline_" in n.get("id", "")]
    if not drop_paths:
        drop_paths = [next(n.iter(NS+"path")) for n in tree.iter(NS+"g") if "water_outline_" in n.get("id", "")]
    drop_extents = [np.ptp(numbers(p.get("d")).reshape(-1,2),axis=0) for p in drop_paths]
    checks["seven_water_icons_identical_dimensions"] = len(drop_extents)==7 and np.allclose(drop_extents,drop_extents[0],atol=1e-5)
    checks["water_outline_weights_match"] = all("stroke-width: 0.65" in p.get("style", "") for p in drop_paths)
    for letter in ["f","i","j"]:
        group = groups["panel_"+letter]
        nodes = list(group.iter())
        scatters = [(idx,n) for idx,n in enumerate(nodes) if f"bar_scatter_{letter}_" in n.get("id", "")]
        errorbars = [(idx,n) for idx,n in enumerate(nodes) if f"errorbar_{letter}_" in n.get("id", "")]
        checks[letter+"_scatter_alpha_matches_j"] = bool(scatters) and all("fill-opacity: 0.72" in use.get("style", "") for _,n in scatters for use in n.iter(NS+"use"))
        checks[letter+"_errorbars_above_all_scatter_points"] = bool(errorbars) and min(idx for idx,_ in errorbars)>max(idx for idx,_ in scatters)
        by_id = {n.get("id"):n for n in group.iter() if n.get("id")}
        offset = numbers(group.get("transform"))[:2]
        bar_centers = []
        for node in group.iter(NS+"g"):
            if not node.get("id", "").startswith(f"final_{letter}_patch_"):
                continue
            for path in node.findall(NS+"path"):
                if re.search(r"fill: #[0-9a-f]{6}", path.get("style", "")):
                    points = numbers(path.get("d")).reshape(-1,2)
                    bar_centers.append((points[:,0].min()+points[:,0].max())/2+offset[0])
        bar_centers.sort()
        brackets = []
        for idx,row in enumerate(manifest["significance_layout"][letter]["placements"]):
            node = by_id[f"final_{letter}_sig_{letter}_bracket_{idx}"]
            vertices = numbers(next(node.iter(NS+"path")).get("d")).reshape(-1,2)+offset
            checks[f"{letter}_bracket_{idx}_consistent_line_weight"] = "stroke-width: 0.65" in next(node.iter(NS+"path")).get("style", "")
            brackets.append(vertices)
            compared = [bar_centers[int(category)] for category in row["groups"]]
            checks[f"{letter}_bracket_{idx}_has_centered_anchors"] = bool(np.allclose(vertices[[0,3],0],compared,atol=1e-5))
            checks[f"{letter}_star_{idx}_unchanged"] = row["stars"]==manifest["significance_layout"][letter]["original_comparisons_and_star_tokens"][idx][2]
        distances = [abs(brackets[n][1,1]-brackets[n-1][1,1])/MM for n in range(1,len(brackets))]
        checks[letter+"_actual_bracket_pitch"] = np.allclose(distances,1.8,atol=1e-5)
        rows = manifest["significance_layout"][letter]["placements"]
        checks[letter+"_visible_glyphs_clear_brackets"] = all(r["visible_star_height_mm"]+.3+.3 < 1.8 for r in rows)
    with fitz.open(stem.with_suffix(".pdf")) as doc:
        page = doc[0]
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b for line in b["lines"] for s in line["spans"]]
        checks["one_pdf_page"] = len(doc)==1
        checks["pdf_bound_to_current_svg"] = doc.metadata.get("subject")=="Figure 1 source SVG SHA256: "+sha(stem.with_suffix(".svg"))
        checks["pdf_page_dimensions"] = np.allclose([page.rect.width/MM,page.rect.height/MM],[180,196],atol=.2)
        checks["one_pdf_raster_image"] = len(page.get_images())==1
        checks["no_text_outside_page"] = all(page.rect.contains(fitz.Rect(s["bbox"])) for s in spans)
        checks["arial_fonts"] = {s["font"] for s in spans}<={"ArialMT","Arial-BoldMT"}
        checks["fixed_physical_font_sizes"] = all(min(abs(s["size"]-v) for v in [5.1,6.5,9])<.025 for s in spans)
        checks["timeline_labels_and_rest"] = all(f"Days {day}–{day+1}" in page.get_text() for day in [1,91,181,271,361]) and page.get_text().count("Rest")==4
        checks["plain_naive_spelling"] = "Naive" in page.get_text() and "Naïve" not in page.get_text()
        drawings = page.get_drawings()
        brackets = [d for d in drawings if len(d["items"])==3 and all(item[0]=="l" for item in d["items"]) and abs(d["items"][0][1].x-d["items"][0][2].x)<.01 and abs(d["items"][1][1].y-d["items"][1][2].y)<.01 and abs(d["items"][2][1].x-d["items"][2][2].x)<.01]
        checks["pdf_ten_brackets_same_weight"] = len(brackets)==10 and all(abs(d["width"]-.65)<.01 for d in brackets)
        drops = [d for d in drawings if d.get("fill") is not None and np.allclose(d["fill"],np.array([54,163,255])/255,atol=.001) and (d["rect"].y0>65*MM or d["rect"].x1<61*MM)]
        checks["pdf_water_icon_sizes_match"] = len(drops)==7 and np.allclose([[d["rect"].width,d["rect"].height] for d in drops],[drops[0]["rect"].width,drops[0]["rect"].height],atol=.01)
    failed = [k for k,v in checks.items() if not v]
    result = {"status":"PASS" if not failed else "FAIL","checks_passed":sum(checks.values()),"checks_total":len(checks),"checks":checks,"validator":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)}}
    (OUT/"VALIDATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    hashes["VALIDATION.json"] = sha(OUT/"VALIDATION.json")
    (OUT/"OUTPUT_HASHES.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":f"{sum(checks.values())}/{len(checks)}","failed":failed},indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
