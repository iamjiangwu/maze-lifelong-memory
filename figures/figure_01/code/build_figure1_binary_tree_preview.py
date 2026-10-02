"""Render Figure 1 from the included data and binary-tree geometry."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT
DEFAULT=Path("figure1_output")
PANEL_A_SHIFT_MM=-1.5
TITLE_MAP={"Day1":"Day 1","Day91":"Day 91","Day2":"Day 2","Day181":"Day 181"}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def tree_source(output):
    source = BASE / "source_code/tree_geometry.py"
    tree = load("figure1_tree_geometry", source)
    geometry = dict(tree.GEOMETRY)
    positions = tree._horizontal_binary_tree_positions(geometry["tree_depth"], geometry["tree_width"], geometry["tree_height"])
    edges=[]
    for level in range(geometry["tree_depth"]-1):
        for index in range(2**level):
            for child in [(level+1,2*index),(level+1,2*index+1)]:
                edges.append([level,index,*positions[(level,index)],*child,*positions[child]])
    with (output/"tree_edges.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["parent_level","parent_index","x0","y0","child_level","child_index","x1","y1"]);w.writerows(edges)
    return geometry,positions,edges,{"source_code/tree_geometry.py": sha(source)}

def build(output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    frozen=output/"frozen_source"
    if not frozen.exists():shutil.copytree(BASE/"frozen_source",frozen)
    geometry,positions,edges,reference_hashes=tree_source(output)
    base=load("tree_preview_ij_base",BASE/"source_code/build_figure1_ij_spacing.py")
    original_load=base.load
    audit={"tree_geometry_from_figure5a":geometry,"tree_edge_count":len(edges),"leaf_count":64,"tree_color":"black","tree_text_count":0}

    def scoped_load(name,path):
        module=original_load(name,path)
        if name=="ij_review_frozen_entrypoint":
            native_load=module.load
            def native(name,path):
                m=native_load(name,path)
                if name=="figure1_approved_layout":
                    m.SLOTS_MM["b"]=(75.5,3.5,103.5,47)
                    old_panel=m.panel_at_final_size
                    def panel(preview,fig,letter):
                        if letter=="e":
                            for ax in fig.axes:ax.set_title(TITLE_MAP[ax.get_title()])
                        if letter in {"e","h"}:
                            for ax in fig.axes:ax.set_title(ax.get_title(),pad=2.8)
                        result=old_panel(preview,fig,letter)
                        if letter in {"b","e","h"}:
                            renderer=fig.canvas.get_renderer()
                            values=[]
                            for ax in fig.axes:
                                if not ax.get_title():continue
                                axis=ax.get_window_extent(renderer)
                                title=ax.title.get_window_extent(renderer)
                                values.append(dict(title=ax.get_title(),gap_pt=(title.y0-axis.y1)*72/fig.dpi,font_size_pt=ax.title.get_fontsize(),axis_bounds_pixels=list(axis.bounds),title_bounds_pixels=list(title.bounds)))
                            audit.setdefault("day_title_spacing",{})[letter]=values
                        return result
                    m.panel_at_final_size=panel
                    old_maze=m.maze_image
                    def maze(root,image,bounds):
                        old_maze(root,image,bounds)
                        for gid in ["panel_a","a_water_port"]:
                            element=next(n for n in root if n.get("id")==gid)
                            element.set("transform",f"translate({PANEL_A_SHIFT_MM*m.MM} 0) "+element.get("transform",""))
                        # Exact Figure5 geometry; re-rendered at physical size, with no text.
                        tw,th=17.2*.8,43.0
                        fig=plt.figure(figsize=(tw/25.4,th/25.4))
                        ax=fig.add_axes([0,0,1,1]);ax.axis("off")
                        for edge in edges:ax.plot([.8*edge[2],.8*edge[6]],[-edge[3],-edge[7]],color="black",lw=.6,solid_capstyle="round")
                        ax.plot([-.8*geometry["root_line_length"],0],[0,0],color="black",lw=.6,solid_capstyle="round")
                        leaves=np.array([positions[(6,i)] for i in range(64)])
                        ax.scatter(.8*leaves[:,0],-leaves[:,1],s=1.5,color="black",edgecolors="none")
                        from ca_analysis.maze import NewMaze
                        maze_model=NewMaze(6)
                        route=[94]
                        while route[-1]!=0:route.append(int(maze_model.pa[route[-1]]))
                        route=route[::-1]
                        decision_rows=list(csv.DictReader((frozen/"plot_ready/figure1_panel_a_decision_points.csv").open()))
                        assert route[:-1]==[int(r["run"]) for r in decision_rows]
                        assert all(child in maze_model.ch[parent] for parent,child in zip(route[:-1],route[1:]))
                        # Raw run IDs do not encode relative left/right turns. Order
                        # children from the physical incoming/outgoing directions.
                        endpoints=np.array([[maze_model.xc[r[-1]],maze_model.yc[r[-1]]] for r in maze_model.ru],float)
                        slots={0:(0,0)};turns={0:"root"}
                        for parent in range(63):
                            level,index=slots[parent]
                            if parent==0:
                                first=maze_model.ru[0][0]
                                incoming=endpoints[parent]-[maze_model.xc[first],maze_model.yc[first]]
                            else:incoming=endpoints[parent]-endpoints[int(maze_model.pa[parent])]
                            assigned=[]
                            for child in maze_model.ch[parent]:
                                child=int(child);outgoing=endpoints[child]-endpoints[parent]
                                cross=incoming[0]*outgoing[1]-incoming[1]*outgoing[0]
                                assert cross!=0 and np.dot(incoming,outgoing)==0
                                # Maze y increases downward. Match the supplied
                                # tree convention: left turn takes the lower fork.
                                side=0 if cross<0 else 1
                                turns[child]="L" if side==0 else "R"
                                slots[child]=(level+1,2*index+side);assigned.append(side)
                            assert sorted(assigned)==[0,1]
                        assert len(set(slots.values()))==127
                        display_positions={node:(.8*positions[slot][0],-positions[slot][1]) for node,slot in slots.items()}
                        route_xy=np.array([display_positions[node] for node in route])
                        ax.plot(route_xy[:,0],route_xy[:,1],color="#2789cf",lw=1.0,solid_capstyle="round",solid_joinstyle="round",zorder=5)
                        ax.scatter(*route_xy[-1],s=2.5,color="#2789cf",edgecolors="none",zorder=6)
                        m.add_standard_water_drop(ax,tuple(route_xy[-1]),ax.transData,"tree_reward",offset=(4.5,0))
                        terminal=maze_model.ru[94][-1]
                        assert (int(maze_model.xc[terminal]),int(maze_model.yc[terminal]))==(14,6)
                        audit["reward_route_nodes"]=route
                        audit["reward_route_turns"]=[turns[node] for node in route[1:]]
                        assert audit["reward_route_turns"]==["L","R","R","L","R","L"]
                        audit["tree_child_order"]="Physical left/right turns, not raw run-ID order; left fork lower, right fork upper"
                        audit["reward_endpoint_node"]=94
                        audit["reward_endpoint_maze_xy"]=[14,6]
                        audit["tree_vertical_reflection"]=-1
                        audit["reward_route_color"]="#2789cf"
                        audit["reward_icon"]="Same 3mm vector water-drop helper as Figure1a/d/e/h; 4.5pt right offset"
                        with (output/"reward_tree_nodes.csv").open("w",newline="") as f:
                            writer=csv.writer(f);writer.writerow(["node","parent","display_level","display_index","incoming_turn","display_x","display_y","maze_x","maze_y","on_reward_route","is_reward_node"])
                            for node,xy in sorted(display_positions.items()):writer.writerow([node,int(maze_model.pa[node]),*slots[node],turns[node],*xy,*endpoints[node],int(node in route),int(node==94)])
                        ax.set_xlim(.8*(-geometry["root_line_length"]-.08),.8*(geometry["tree_height"]+.10))
                        extent=geometry["tree_width"]/2+.12
                        ax.set_ylim(-extent,extent);ax.set_aspect("equal")
                        buffer=io.StringIO()
                        with matplotlib.rc_context({"savefig.bbox":None,"savefig.pad_inches":0,"svg.hashsalt":"figure1_black_tree"}):
                            fig.savefig(buffer,format="svg",transparent=True,bbox_inches=None,metadata={"Date":None})
                        plt.close(fig)
                        element=ET.fromstring(buffer.getvalue())
                        preview=sys.modules["record_local_main1_preview"]
                        preview._prefix_svg_ids(element,"black_tree_")
                        group=ET.SubElement(root,m.tag("g"),{"id":"panel_a_binary_tree","transform":f"translate({(58.5+PANEL_A_SHIFT_MM)*m.MM} {5*m.MM})"})
                        for child in element:
                            if child.tag!=m.tag("metadata"):group.append(child)
                        audit["tree_slot_mm"]=[58.5+PANEL_A_SHIFT_MM,5,tw,th]
                        audit["panel_a_artwork_shift_mm"]=PANEL_A_SHIFT_MM
                        audit["tree_horizontal_coordinate_scale"]=.8
                        audit["tree_linewidth_pt"]=.6
                    m.maze_image=maze
                    def trajectories(preview,ed,data,cases,name):
                        plt.rcParams.update({"font.family":"Arial","font.size":6.5,"svg.fonttype":"none","pdf.fonttype":42})
                        width,height=103.5,47
                        fig=plt.figure(figsize=(width/25.4,height/25.4))
                        norm=Normalize(0,float(data.elapsed_minutes.max()))
                        counts={};artists={}
                        for i,(session,title) in enumerate(cases):
                            # Position the two trajectory axes in millimetre-derived slots.
                            ax=fig.add_axes([i*48.2/width,(height-3-49.25*15.2/17.8)/height,49.25/width,(49.25*15.2/17.8)/height])
                            rows=data.loc[data.nickname.eq(session)]
                            ed.draw_trajectory(ax,rows,title,norm)
                            ax.lines[0].set_gid(f"maze_wall_b_{i}")
                            ax.set_ylim(14.6,-.6);ax.set_title(title,fontsize=6.5,pad=2)
                            counts[session]=int(rows.draw_to_next.iloc[:-1].sum())
                            collection=ax.collections[0]
                            payload=np.asarray(collection.get_segments()).tobytes()+np.asarray(collection.get_array()).tobytes()
                            artists[session]="sha256:"+hashlib.sha256(payload).hexdigest()
                        cax=fig.add_axes([97.05/width,(height-44)/height,1.1/width,39.8/height])
                        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="viridis"),cax=cax,orientation="vertical")
                        cb.solids.set_rasterized(False);cb.set_ticks([0,norm.vmax]);cb.set_ticklabels(["0",f"{norm.vmax:.0f}"])
                        cb.ax.tick_params(labelsize=6.5,length=1.5,pad=1.5);cb.outline.set_linewidth(.4)
                        cb.set_label("Time in maze (min)",fontsize=6.5,labelpad=-8.5)
                        audit["trajectory_artists_hashes"]=artists
                        audit["trajectory_color_range"]=[0,norm.vmax]
                        audit["colorbar"]="vertical on the right; title labelpad -8.5 pt"
                        result=m.panel_at_final_size(preview,fig,"b")
                        audit["trajectory_axes_mm"]=result["axes_final_mm"]
                        return result,counts
                    m.trajectories=trajectories
                return m
            module.load=native
        return module
    base.load=scoped_load
    base.build(output,record=None)
    audit.update(status="PASS", source_manifest_sha256=sha(frozen/"SOURCE_INPUTS.csv"), reference_sources=reference_hashes)
    dump(output/"PREVIEW_VALIDATION.json",audit)
    shutil.copy2(__file__,output/"source_code"/Path(__file__).name)
    (output/"README.md").write_text("# Figure 1 source build\n\nThe binary tree highlights maze nodes 0-1-4-10-22-46-94 in blue, with the water icon at node 94. Each fork follows physical left/right turns: L-R-R-L-R-L. The terminal maze coordinate is (14,6). reward_tree_nodes.csv records all 127 node assignments.\n\nPREVIEW_VALIDATION.json records source hashes, segment counts and display geometry. Reproduce from the publication repository with `python reproduce.py --figure 1 --output-dir OUTPUT`.\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=DEFAULT)
    build(parser.parse_args().output_dir)
