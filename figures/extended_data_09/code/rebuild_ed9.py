"""Render the author's three-row ED9 revision from frozen numerical sources.



The approved Level population-vector distribution leads row 1. The joint

place/level cloud moves between the Level control and within-level control in

row 2. All panel dimensions are native; row 2 is 6 mm shorter than its predecessor.

No accepted artwork is read by this builder.

"""

from __future__ import annotations



import argparse

import hashlib

import json

from pathlib import Path



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from matplotlib.patches import Rectangle

from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from matplotlib.transforms import Bbox

import numpy as np

import pandas as pd

from scipy.stats import t as student_t

from PIL import Image



WIDTH, HEIGHT = 183.0, 132.0

LEVEL, PLACE, GRAY = "#7976A2", "#B95A58", "#7F7F7F"

EXPERIENCE = dict(zip(["naive", "3mo", "6mo", "9mo", "12mo"],

                      ["#BDB9D6", "#A19CC3", "#8A85B2", "#6F6B99", "#4E4A75"]))

FONT, TICK, LETTER = 6.5, 6.0, 10.5

PANEL_BOXES = {

    "a": (0,0,35,45), "b": (35,0,28,45), "c": (63,0,26,45),

    "d": (89,0,25,45), "e": (114,0,41,45), "f": (155,0,28,45),

    "g": (0,45,56,41), "h": (56,45,35,41), "i": (91,45,36,41),

    "j": (127,45,56,41),

    "k": (0,86,37,46), "l": (37,86,36,46), "m": (73,86,37,46),

    "n": (110,86,36.5,46), "o": (146.5,86,36.5,46),

}

# old_panel refers to the immediately preceding 14-panel ED9 review.

CROSSWALK = [

    ('a','new','Level PV correlation distribution',1),

    ('b','a','Population-vector mean correlation shuffle-null distribution',1),

    ('c','b','Mean Level correlation shuffle-null distribution',1),

    ('d','c','Level decoding accuracy versus neuron-ID shuffle',1),

    ('e','d','Preferred-level transition matrix',1),

    ('f','e','Level shift distribution',1),

    ('g','g','Horizontal and vertical eight-bin schematics only',2),

    ('h','h','Horizontal / ordinary Level / vertical comparison',2),

    ('i','f','Place versus level map correlation',2),

    ('j','i','Within-level shuffle pair comparison and null histogram',2),

    ('k','j','Level correlation across time gaps',3),

    ('l','k','Consecutive-gap persistence',3),

    ('m','l','Within-session learning curves',3),

    ('n','m','First-30-minute Level alignment versus cycles',3),

    ('o','n','Long-gap Level correlation versus cycles',3),

]





def sha(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()





def write_json(path, value):

    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")





def jitter(keys, width):

    return np.array([((int(hashlib.sha256(str(k).encode()).hexdigest()[:8], 16) % 10001) / 10000 - .5) * 2 * width for k in keys])





def stars(p):

    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."





def bracket(ax, x1, x2, y, label):

    ax.plot([x1, x1, x2, x2], [y-.018, y, y, y-.018], color="black", lw=.65, clip_on=False, zorder=10)

    ax.text((x1+x2)/2, y+.012, label, ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")





def fit_band(x, y, groups, xx, intercept, slope):

    design = np.column_stack([np.ones(len(x)), x])

    bread = np.linalg.inv(design.T @ design)

    resid = y - intercept - slope*x

    unique = np.unique(groups)

    meat = sum(np.outer(design[groups == g].T @ resid[groups == g], design[groups == g].T @ resid[groups == g]) for g in unique)

    cov = len(unique)/(len(unique)-1) * (len(x)-1)/(len(x)-2) * bread @ meat @ bread

    xxdesign = np.column_stack([np.ones(len(xx)), xx])

    se = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", xxdesign, cov, xxdesign)))

    ci = student_t.ppf(.975, len(unique)-1) * se

    return intercept+slope*xx-ci, intercept+slope*xx+ci





def build(inputs: Path, output: Path):

    output.mkdir(parents=True, exist_ok=True)

    data_dir = inputs / "source_data/frozen"

    comparison_dir = inputs / "source_data/level_comparison"

    input_audit = json.loads((inputs / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))

    assert all(sha(data_dir / row["frozen_file"]) == row["sha256"] for row in input_audit)

    comparison_manifest = json.loads((comparison_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    assert comparison_manifest["status"] == "PASS"

    assert all(sha(comparison_dir / name) == row["sha256"]

               for name, row in comparison_manifest["outputs"].items())

    first30_dir = inputs / "source_data/first30_addition"

    first30_manifest = json.loads((first30_dir / "manifest.json").read_text(encoding="utf-8"))

    assert first30_manifest["status"] == "PASS"

    assert all(sha(first30_dir / name) == row["sha256"]

               for name, row in first30_manifest["outputs"].items())

    pv_shuffle_dir = inputs / "source_data/population_vector_shuffle"

    pv_shuffle_manifest = json.loads((pv_shuffle_dir / "manifest.json").read_text(encoding="utf-8"))

    assert pv_shuffle_manifest["status"] == "PASS"

    assert all(sha(pv_shuffle_dir / name) == row["sha256"]

               for name,row in pv_shuffle_manifest["outputs"].items())

    distribution_dir=inputs/"source_data/population_vector_distribution"

    distribution_hashes=json.loads((distribution_dir/"OUTPUT_HASHES.json").read_text(encoding="utf-8"))

    assert all(sha(distribution_dir/name)==value for name,value in distribution_hashes.items() if Path(name).suffix.lower() not in {".pdf", ".svg", ".png", ".tif", ".tiff", ".html"})

    read = lambda name: pd.read_csv(data_dir / name)

    plt.rcParams.update({"font.family": "Arial", "font.size": FONT, "axes.labelsize": FONT,

        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.linewidth": .65,

        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,

        "svg.fonttype": "none", "svg.hashsalt": "ed9-three-row-level-pv",

        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True})

    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), dpi=150)

    axes = {}

    evidence = {}

    def axis(name, x, top, w, h, *, off=False):

        ax = fig.add_axes([x/WIDTH, 1-(top+h)/HEIGHT, w/WIDTH, h/HEIGHT])

        axes[name] = ax

        ax.set_gid(name)

        ax.tick_params(width=.65, length=2, pad=1.5)

        ax.xaxis.labelpad = 2.0

        ax.yaxis.labelpad = 2.0

        if off:

            ax.axis("off")

        return ax

    for label, (x, top, w, h) in PANEL_BOXES.items():

        fig.text((x+.6)/WIDTH, 1-(top+1)/HEIGHT, label, fontsize=LETTER, fontweight="bold", va="top", gid=f"panel_letter_{label}")



    # New distribution panel: the approved standalone Figure2d level counterpart.

    ax=axis("a_population_vector_distribution",8,10,25,26)

    with np.load(distribution_dir/"plot_data.npz") as source:

        distribution_obs=source["obs"].astype(float)

        distribution_edges=source["hist_edges"].astype(float)

        distribution_null_counts=source["shuffle_hist"].astype(int)

    widths=np.diff(distribution_edges);centers=(distribution_edges[:-1]+distribution_edges[1:])/2

    distribution_obs_counts=np.histogram(distribution_obs,bins=distribution_edges)[0]

    distribution_data_density=distribution_obs_counts/(len(distribution_obs)*widths)

    distribution_null_density=distribution_null_counts/(distribution_null_counts.sum()*widths)

    distribution_bars=[ax.bar(centers,distribution_null_density,width=widths,color="#8a8a8a",alpha=.62,edgecolor="none",zorder=1),

                       ax.bar(centers,distribution_data_density,width=widths,color=LEVEL,alpha=.72,edgecolor="none",zorder=2)]

    ax.set(xlim=(-.45,1),ylim=(0,max(distribution_null_density.max(),distribution_data_density.max())*1.4),

           xlabel="Level PV correlation",ylabel="Density",xticks=[0,.5,1])

    ax.yaxis.set_major_locator(MaxNLocator(3))

    distribution_labels=[ax.text(.05,.97,"Shuffle",transform=ax.transAxes,ha="left",va="top",fontsize=FONT,color="#6f6f6f",fontweight="bold"),

                         ax.text(.62,.37,"Data",transform=ax.transAxes,ha="left",va="center",fontsize=FONT,color=LEVEL,fontweight="bold")]

    evidence["a"]={"observations":len(distribution_obs),"pairs":36,"animals":14,

                   "null_values":int(distribution_null_counts.sum()),"bin_edges":distribution_edges.tolist(),

                   "data_counts":distribution_obs_counts.tolist(),"shuffle_counts":distribution_null_counts.tolist(),

                   "pooled_observed_mean":float(distribution_obs.mean()),"weighting":"equal supported pair-level observations"}



    # Population-vector neuron-ID shuffle, with the same histogram grammar.

    ax=axis("b_population_vector_null",43,10,17,26)

    pv_null=pd.read_csv(pv_shuffle_dir/"null_draws.csv").mean_correlation.to_numpy(float)

    pv_summary=json.loads((pv_shuffle_dir/"summary.json").read_text(encoding="utf-8"))

    pv_observed=float(pv_summary["observed_mean"])

    pv_density,pv_edges,_=ax.hist(pv_null,bins=80,density=True,color=GRAY,edgecolor="#555555",lw=.5,alpha=.75)

    ax.axvline(pv_observed,color=LEVEL,lw=1.2,ls="--")

    ax.text(.18,.77,"Shuffle",transform=ax.transAxes,color=GRAY,fontsize=FONT)

    ax.text(.55,.52,"Data",transform=ax.transAxes,color=LEVEL,fontsize=FONT)

    ax.set(xlim=(-.035,.44),xticks=[0,.2,.4],xlabel="Mean level PV\ncorrelation",ylabel="Density")

    evidence["b"]={"null_draws":len(pv_null),"observed_mean":pv_observed,"null_mean":float(pv_null.mean()),"density":pv_density.tolist(),"bin_edges":pv_edges.tolist(),"weighting":"equal mean over36pair-specific supported-level diagonal means","null":"saved within-pair target-neuron-ID permutations shared across levels"}



    # Top row: the two transferred Figure5 controls precede retained ED9 plots.

    ax = axis("c_mean_null",70,10,16,26)

    with np.load(data_dir/"figure5_mean_null.npz") as saved:

        null=np.asarray(saved["null"],dtype=float);observed=float(np.asarray(saved["obs"],dtype=float).ravel()[0])

    null=null[np.isfinite(null)]

    density,bins,_=ax.hist(null,bins=80,density=True,color=GRAY,edgecolor="#555555",lw=.5,alpha=.75)

    c_observed_line=ax.axvline(observed,color=LEVEL,lw=1.2,ls="--")

    ax.text(.20,.77,"Shuffle",transform=ax.transAxes,color=GRAY,fontsize=FONT)

    c_data_label=ax.annotate("Data",xy=(observed,.52),xycoords=("data","axes fraction"),

                             xytext=(-4,0),textcoords="offset points",ha="right",

                             color=LEVEL,fontsize=FONT)

    ax.set(xlim=(0,.3),xticks=[0,.1,.2,.3],xlabel="Mean level map\ncorrelation",ylabel="Density")

    evidence["c"]={"source_panel":"Figure5 d","null_draws":len(null),"observed":observed,"density":density.tolist(),"bin_edges":bins.tolist()}



    ax=axis("d_decoding",96,10,16,26)

    decoder=read("figure5_decoder_points.csv")

    dec_summary=read("figure5_decoder_summary.csv").set_index("group")

    dec_stats=json.loads((data_dir/"figure5_decoder_stats.json").read_text())

    dec_wide=decoder.pivot(index="pair_id",columns="group",values="decoding_accuracy")

    for row in dec_wide.itertuples():

        ax.plot([0,1],[row.Data,row.Shuffle],color="#8a8a8a",alpha=.28,lw=.65,zorder=1)

    rng=np.random.default_rng(20260420)

    for x,group,color in [(0,"Data",LEVEL),(1,"Shuffle",GRAY)]:

        points=decoder.loc[decoder.group.eq(group),"decoding_accuracy"].to_numpy()

        assert np.isclose(points.mean(),dec_summary.loc[group,"mean"],atol=1e-15)

        ax.bar(x,dec_summary.loc[group,"mean"],width=.66,color=color,edgecolor="none",alpha=.78,zorder=2)

        ax.scatter(x+rng.normal(0,.03,len(points)),points,s=13,color=color,edgecolors="none",alpha=.65,zorder=3)

    bracket(ax,0,1,.610,stars(dec_stats["cluster_ols"]["p_greater"]))

    ax.set(ylim=(0,.66),yticks=[0,.2,.4,.6],xlim=(-.5,1.5),xticks=[0,1],xticklabels=["Data","Shuffle"],ylabel="Decoding accuracy")

    evidence["d"]={"source_panel":"Figure5 e","pairs":len(dec_wide),"animals":int(decoder.animal.nunique()),"means":dec_summary["mean"].to_dict(),"p_greater":dec_stats["cluster_ols"]["p_greater"]}



    ax = axis("e_transition",121,10,26,26)

    cells = read("transition.csv")

    matrix = cells.pivot(index="test_pref_level", columns="train_pref_level", values="observed_minus_null_conditional_probability").sort_index().sort_index(axis=1).to_numpy()

    np.fill_diagonal(matrix, np.nan)

    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("#F2F2F2")

    im = ax.imshow(matrix, origin="lower", cmap=cmap, norm=TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=.1), interpolation="nearest")

    ax.set(xticks=range(7), yticks=range(7), xlabel="Day i level", ylabel="Day i+90 level")

    ax.tick_params(length=0)

    for spine in ax.spines.values(): spine.set_visible(False)

    cax = axis("e_colorbar",148.5,12,1.3,22)

    cb = fig.colorbar(im, cax=cax, ticks=[-.1, 0, .1]); cb.ax.tick_params(length=1.5, pad=1)

    cb.ax.set_yticklabels(["\u22120.1", "0", "0.1"])

    cb.outline.set_linewidth(.45)

    ax.set_title("\u0394 conditional probability", fontsize=FONT, pad=3)

    evidence["e"] = {"shape": list(matrix.shape), "off_diagonal_max_abs": float(np.nanmax(abs(matrix))), "levels": list(range(7))}



    ax = axis("f_shift",165,10,16,26)

    shift = read("shift.csv")

    ax.plot(shift.distance, shift.observed_fraction_shifted, color=LEVEL, marker="o", ms=3, lw=1.1, label="Observed")

    ax.errorbar(shift.distance, shift.null_mean_fraction_shifted, yerr=shift.null_sd_fraction_shifted, color=GRAY, marker="o", mfc="white", mec=GRAY, ms=2.7, lw=.9, ls="--", capsize=1.4, label="ID null")

    ax.set(xticks=range(1,7), xlabel="Level shift distance", ylabel="Fraction of shifted cells", ylim=(.04,.365))

    ax.xaxis.label.set_x(.45625)

    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.4, handletextpad=.4, borderaxespad=.1, labelspacing=.3)

    evidence["f"] = {"observed": shift.observed_fraction_shifted.tolist(), "null": shift.null_mean_fraction_shifted.tolist()}



    ax = axis("i_joint_cloud",101,51,24,26)

    cloud = read("joint_cloud.csv"); cloud = cloud[cloud.continuous_cloud_included.astype(bool)]

    ax.scatter(cloud.place_map_corr, cloud.level_corr, s=2, color=LEVEL, alpha=.035, edgecolors="none", rasterized=True)

    ax.plot([-.65,1],[-.65,1],color="#A6A6A6",lw=.65,ls="--")

    ax.set(xlim=(-.65,1.02),ylim=(-.65,1.02),xticks=[-.5,0,.5,1],yticks=[-.5,0,.5,1],xlabel="Place map correlation",ylabel="Level map correlation")

    evidence["i"] = {"observations": len(cloud), "pairs": int(cloud.pair_key.nunique())}



    # Middle row: small intact maze controls, Level comparison and former c.

    maze = read("maze_cells.csv"); walls = read("maze_walls.csv")

    blocks = {value:index for index,values in enumerate((range(0,3),range(4,7),range(8,11),range(12,15))) for value in values}

    hgroups=[(0,1),(4,5),(2,3),(6,7),(8,9),(12,13),(10,11),(14,15)]

    vgroups=[(0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)]

    for name,x,groups,title in [("g_horizontal",1.0,hgroups,"Horizontal 8-bin"),("g_vertical",29.0,vgroups,"Vertical 8-bin")]:

        ax = axis(name,x,51,26,26,off=True)

        lookup={block:group for group,member in enumerate(groups) for block in member}

        palette=plt.get_cmap("magma")(np.linspace(0,1,8))

        shown=[]

        for row in maze.itertuples():

            if row.x not in blocks or row.y not in blocks: continue

            group=lookup[blocks[row.x]+4*blocks[row.y]]

            shown.append(group)

            ax.add_patch(Rectangle((row.x-.5,row.y-.5),1,1,facecolor=palette[group],edgecolor="none"))

        ax.plot(walls.x,walls.y,color="black",lw=.5,solid_capstyle="round")

        ax.set(xlim=(-.65,14.65),ylim=(-.65,14.65),aspect="equal")

        ax.set_title(title,fontsize=FONT,pad=2)

        evidence[name] = {"groups": sorted(set(shown)), "included_cells":len(shown), "maze_size_mm":26}



    ax = axis("h_level_controls",64,51,25,26)

    comp=pd.read_csv(comparison_dir/"pair_values.csv")

    order=["horizontal_dumbbell8","level_code","vertical_dumbbell8"]

    wide=comp.pivot(index="pair_key",columns="representation",values="mean_raw_r")[order]

    j=jitter(wide.index,.09)

    for values in wide.to_numpy(): ax.plot([0,1,2],values,color="#D9D9D9",lw=.4,alpha=.7,zorder=1)

    for x,key in enumerate(order):

        ax.scatter(x+j,wide[key],s=9,color=LEVEL if x==1 else GRAY,alpha=.65,edgecolors="none",zorder=2)

        ax.scatter([x],[wide[key].mean()],s=32,facecolors=LEVEL if x==1 else "white",edgecolors=LEVEL if x==1 else GRAY,lw=1,zorder=4)

    stat=pd.read_csv(comparison_dir/"statistics.csv")

    bracket(ax,0,1,.62,stat.iloc[0].stars);bracket(ax,1,2,.72,stat.iloc[1].stars)

    ax.set(xticks=[0,1,2],xticklabels=["Horizontal\n8-bin","Level","Vertical\n8-bin"],ylabel="Pair-mean r",ylim=(-.23,.83),xlim=(-.35,2.35))

    ax.yaxis.set_major_locator(MaxNLocator(5))

    evidence["h"]={"pairs":len(wide),"means":wide.mean().to_dict(),"contrast_p":stat.p_adjusted_bh2.tolist(),"stars":stat.stars.tolist()}



    ax=axis("j_pair",136,51,18,26)

    pairs=read("within_level_pairs.csv").query("scope == 'global'")

    pair_jitter=jitter(pairs.pair_key,.04)

    for row in pairs.itertuples(): ax.plot([0,1],[row.null_raw_mean,row.observed_raw_mean],color="#D9D9D9",lw=.4,alpha=.65,zorder=1)

    ax.scatter(pair_jitter,pairs.null_raw_mean,s=9,facecolor="white",edgecolor=GRAY,lw=.6,alpha=.76,zorder=2)

    ax.scatter(1+pair_jitter,pairs.observed_raw_mean,s=9,color=PLACE,edgecolors="none",alpha=.70,zorder=2)

    means=[pairs.null_raw_mean.mean(),pairs.observed_raw_mean.mean()]

    ax.plot([0,1],means,color="black",lw=1,zorder=3)

    ax.scatter([0],[means[0]],s=32,facecolor="white",edgecolor=GRAY,lw=1,zorder=4)

    ax.scatter([1],[means[1]],s=32,facecolor=PLACE,edgecolor="black",lw=.4,zorder=4)

    cstat=read("within_level_stats.csv").query("metric == 'raw'").iloc[0]

    bracket(ax,0,1,.48,stars(cstat.p_directional))

    ax.set(xlim=(-.18,1.18),ylim=(-.03,.60),xticks=[0,1],xticklabels=["Within-level\nshuffle","Data"],ylabel="Place map correlation")

    ax=axis("j_null",163.3,51,17,26)

    null_draws=pd.read_csv(data_dir/"within_level_nulls.csv",float_precision="round_trip")

    assert null_draws.repeat.is_unique and null_draws.n_pairs.eq(len(pairs)).all()

    nulls=null_draws.set_index("repeat").mean_correlation.sort_index()

    assert len(nulls)==1000 and len(pairs)==36

    counts,edges,_=ax.hist(nulls,bins=45,color="#6F6F6F",alpha=.78,edgecolor="none")

    ax.axvline(means[1],color=PLACE,lw=1.2,ls="--")

    span=max(means[1]-nulls.min(),nulls.max()-nulls.min())

    ax.set(xlim=(nulls.min()-.08*span,means[1]+.10*span),ylim=(0,counts.max()*1.2),xlabel="Place map\ncorrelation",ylabel="Count")

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"));ax.xaxis.set_major_locator(MaxNLocator(2));ax.yaxis.set_major_locator(MaxNLocator(3,integer=True))

    ax.text(.07,.95,"Shuffle",transform=ax.transAxes,ha="left",va="top",color="#6F6F6F",fontsize=6)

    ax.text(.84,.78,"Data",transform=ax.transAxes,ha="right",va="center",color=PLACE,fontsize=6)

    evidence["j"]={"pair_count":len(pairs),"null_repeats":len(nulls),"observed_mean":float(means[1]),"shuffle_mean":float(means[0]),"p_directional":float(cstat.p_directional)}



    # Bottom row: five source-sized plots, including the added first30 panel.

    ax=axis("k_gaps",8,91,27,31)

    gaps=read("gap.csv").sort_values("time_order")

    x=gaps.time_days.to_numpy(float)

    ax.fill_between(x,gaps.observed_ci95_low,gaps.observed_ci95_high,color=LEVEL,alpha=.16,lw=0)

    ax.plot(x,gaps.observed_mean,color=LEVEL,marker="o",ms=3,lw=1.2,label="Data")

    ax.plot(x,gaps.shuffle_mean,color=GRAY,marker="s",mfc=GRAY,ms=2.4,lw=.75,ls=(0,(2,1.4)),label="Shuffle")

    ax.set(xlim=(-8,370),ylim=(-.08,.84),xticks=x,xticklabels=["1","90","180","270","360"],xlabel="Days apart",ylabel="Correlation")

    ax.legend(loc="upper right",fontsize=5.8,handlelength=1.1,handletextpad=.3,borderaxespad=.1,labelspacing=.2)

    evidence["k"]={"gap_days":x.tolist(),"means":gaps.observed_mean.tolist()}



    ax=axis("l_consecutive",45,91,25.5,31)

    consecutive=read("consecutive_points.csv").query("metric == 'level_map_correlation'").reset_index(drop=True)

    # The final source contains only rewarded session chains, for both the
    # point cloud and trend. No upstream cohort-selection table is needed.
    fit=consecutive

    ax.scatter(consecutive.prev_gap_correlation,consecutive.next_gap_correlation,s=4,color=LEVEL,alpha=.035,edgecolors="none",rasterized=True)

    edges=np.linspace(-1,1,14);centers=(edges[:-1]+edges[1:])/2;trend=[]

    for i in range(13):

        selected=fit[(fit.prev_gap_correlation>=edges[i]) & ((fit.prev_gap_correlation<=edges[i+1]) if i==12 else (fit.prev_gap_correlation<edges[i+1]))]

        trend.append(selected.next_gap_correlation.mean() if len(selected)>=3 else np.nan)

    keep=np.isfinite(trend);ax.plot(centers[keep],np.asarray(trend)[keep],color=LEVEL,lw=1.3)

    p=float(read("consecutive_stats.csv").iloc[0].q_bh_two_sided)

    ax.text(.08,.91,stars(p),transform=ax.transAxes,ha="left",va="top",fontsize=8,fontweight="bold")

    ax.set(xlim=(-1,1),ylim=(-1,1),xticks=[-1,0,1],yticks=[-1,0,1],xlabel="Previous gap correlation",ylabel="Next gap correlation")

    evidence["l"]={"display_rows":len(consecutive),"fit_rows":len(fit),"q_bh":p,"trend_x":centers[keep].tolist(),"trend_y":np.asarray(trend)[keep].tolist()}



    ax=axis("m_learning",81,91,28,31)

    learning=read("learning.csv");learning=learning[learning.passes_support_mask.astype(bool)]

    learning_rows=[]

    for group,color in EXPERIENCE.items():

        d=learning[learning.experience_group.eq(group)].sort_values("x_plot_minutes")

        x=d.x_plot_minutes.to_numpy(float)

        y=d["mean"].rolling(5,center=True,min_periods=1).mean().to_numpy()

        sem=d["sem"].fillna(0).rolling(5,center=True,min_periods=1).mean().to_numpy()

        support=d.n_animals.to_numpy()>=3

        ax.fill_between(x,y-sem,y+sem,where=support,color=color,alpha=.14,lw=0)

        segments=[[[x[i],y[i]],[x[i+1],y[i+1]]] for i in range(len(x)-1) if support[i] and support[i+1] and np.isfinite([x[i],x[i+1],y[i],y[i+1]]).all()]

        lines=LineCollection(segments,colors=[color],lw=1.15,label=group);lines.set_capstyle("round");ax.add_collection(lines)

        learning_rows.extend(dict(group=group,time=a,mean=b,sem=c,supported=bool(s)) for a,b,c,s in zip(x,y,sem,support))

    ax.set(xlim=(25,185),ylim=(.10,.86),xticks=[50,100,150],yticks=[.2,.4,.6,.8],xlabel="Time in maze (min)",ylabel="Level map correlation")

    ax.legend(loc="upper left",fontsize=5.5,ncol=2,handlelength=.8,columnspacing=.6,handletextpad=.2,borderaxespad=.05,labelspacing=.2)

    pd.DataFrame(learning_rows).to_csv(output/"learning_display_values.csv",index=False)

    evidence["m"]={"source_rows":len(learning),"smoothing":"centered 5-point mean","minimum_animals":3}



    ax=axis("n_first30",118,91,26,31)

    first30=pd.read_csv(first30_dir/"first30_points.csv").query("metric == 'level_map_correlation'").reset_index(drop=True)

    firststat=pd.read_csv(first30_dir/"first30_stats.csv").iloc[0]

    assert len(first30)==59 and first30.animal.nunique()==22

    keys=first30.animal.astype(str)+"|"+first30.experience_index.astype(str)

    jj=jitter(keys,.07)

    for exp,group in enumerate(EXPERIENCE):

        selected=first30.experience_index.eq(exp)

        ax.scatter(first30.loc[selected,"experience_index"]+jj[selected],first30.loc[selected,"value"],s=12,color=EXPERIENCE[group],edgecolors="none",alpha=.72,zorder=3)

    x=first30.experience_index.to_numpy(float);y=first30.value.to_numpy(float);xx=np.linspace(0,4,100)

    lo,hi=fit_band(x,y,first30.animal.astype(str).to_numpy(),xx,firststat.intercept,firststat.slope)

    ax.fill_between(xx,lo,hi,color=LEVEL,alpha=.16,lw=0,zorder=1)

    ax.plot(xx,firststat.intercept+firststat.slope*xx,color=LEVEL,lw=1.25,zorder=2)

    ax.text(.05,.96,stars(firststat.q_bh_two_sided),transform=ax.transAxes,ha="left",va="top",fontsize=8,fontweight="bold")

    ax.text(.05,.82,f"r={firststat.pearson_r:.2f}",transform=ax.transAxes,ha="left",va="top",fontsize=FONT)

    ax.set(xlim=(-.45,4.45),ylim=(0,.75),xticks=[0,1,2,3,4],yticks=[0,.2,.4,.6],xlabel="No. of cycles",ylabel="Level map correlation")

    ax.set_title("First 30 min",fontsize=FONT,pad=3)

    evidence["n"]={"rows":len(first30),"animals":int(first30.animal.nunique()),"r":float(firststat.pearson_r),"q_bh":float(firststat.q_bh_two_sided),"slope":float(firststat.slope),"estimand":"first 30 min Day i compared with full Day i+1 template"}



    ax=axis("o_cycles",154.5,91,26.5,31)

    cycles=read("cycles.csv").query("metric == 'level_map_correlation'").reset_index(drop=True)

    cycles["pair_key"]=cycles.src+"->"+cycles.dst

    fitstat=read("cycles_stats.csv").iloc[0]

    jj=jitter(cycles.pair_key,.07)

    for exp,group in enumerate(["3mo","6mo","9mo","12mo"],start=1):

        selected=cycles.experience.eq(exp)

        ax.scatter(cycles.loc[selected,"experience"]+jj[selected],cycles.loc[selected,"value"],s=12,color=EXPERIENCE[group],edgecolors="none",alpha=.72,zorder=3)

    x=cycles.experience.to_numpy(float);y=cycles.value.to_numpy(float);xx=np.linspace(1,4,100)

    lo,hi=fit_band(x,y,cycles.animal.astype(str).to_numpy(),xx,fitstat.intercept,fitstat.slope)

    ax.fill_between(xx,lo,hi,color=LEVEL,alpha=.16,lw=0,zorder=1)

    ax.plot(xx,fitstat.intercept+fitstat.slope*xx,color=LEVEL,lw=1.25,zorder=2)

    ax.text(.05,.96,stars(fitstat.q_bh_two_sided),transform=ax.transAxes,ha="left",va="top",fontsize=8,fontweight="bold")

    ax.text(.05,.82,f"r={np.corrcoef(x,y)[0,1]:.2f}",transform=ax.transAxes,ha="left",va="top",fontsize=FONT)

    ax.set(xlim=(.55,4.45),ylim=(-.06,.58),xticks=[1,2,3,4],xlabel="Cycle",ylabel="Level map correlation")

    ax.set_title("Long gap",fontsize=FONT,pad=3)

    evidence["o"]={"pairs":len(cycles),"r":float(np.corrcoef(x,y)[0,1]),"q_bh":float(fitstat.q_bh_two_sided),"slope":float(fitstat.slope)}



    fig.canvas.draw()

    rend=fig.canvas.get_renderer()

    # Measure the rendered text edge against the full dashed-line stroke.

    c_label_box=c_data_label.get_window_extent(rend)

    c_line_x=axes["c_mean_null"].transData.transform((float(c_observed_line.get_xdata()[0]),0))[0]

    c_clearance_pt=(c_line_x-c_label_box.x1)*72/fig.dpi-c_observed_line.get_linewidth()/2

    write_json(output/"PANEL_C_LABEL_CLEARANCE.json",{

        "status":"PASS" if c_clearance_pt>=3 else "FAIL",

        "label":"Data","line_clearance_pt":float(c_clearance_pt),

        "minimum_clearance_pt":3,"position":"right aligned, 4 pt left of observed line"})

    distribution_overlaps=[]

    for label in distribution_labels:

        bounds=label.get_window_extent(rend).padded(fig.dpi/72)

        for bars in distribution_bars:

            for bar in bars:

                if bar.get_height()>0 and bounds.overlaps(bar.get_window_extent(rend)):

                    distribution_overlaps.append(label.get_text())

    write_json(output/"DISTRIBUTION_LABEL_QA.json",{"status":"PASS" if not distribution_overlaps else "FAIL","label_mark_overlaps":distribution_overlaps})

    text_overflow=[]

    for artist in fig.findobj(matplotlib.text.Text):

        if not artist.get_visible() or not artist.get_text(): continue

        b=artist.get_window_extent(rend)

        # Suppress unused tick artists outside explicit axis limits.

        if not artist.get_clip_on() and (b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1):

            text_overflow.append(artist.get_text())

    geometry={name:{"bounds_mm":[float(ax.get_position().x0*WIDTH),float((1-ax.get_position().y1)*HEIGHT),float(ax.get_position().width*WIDTH),float(ax.get_position().height*HEIGHT)],"xlim":list(ax.get_xlim()),"ylim":list(ax.get_ylim())} for name,ax in axes.items()}

    write_json(output/"PANEL_GEOMETRY.json",geometry)

    write_json(output/"PANEL_EVIDENCE.json",evidence)

    pd.DataFrame(CROSSWALK,columns=["new_panel","old_panel","content","row"]).to_csv(output/"PANEL_CROSSWALK.csv",index=False)

    stem=output/"ed9_three_row_level_pv"

    metadata={"Date":None}

    fig.savefig(stem.with_suffix(".svg"),metadata=metadata,facecolor="white",dpi=600)

    fig.savefig(stem.with_suffix(".pdf"),metadata={"CreationDate":None,"ModDate":None},facecolor="white",dpi=600)

    for dpi in (300,600):

        path=output/f"{stem.name}_{dpi}dpi.png";fig.savefig(path,dpi=dpi,facecolor="white")

        with Image.open(path) as im: im.convert("RGB").save(path,dpi=(dpi,dpi))

    with Image.open(output/f"{stem.name}_600dpi.png") as im: im.save(output/f"{stem.name}_600dpi.tiff",compression="tiff_lzw",dpi=(600,600))

    panels=output/"individual_panels";panels.mkdir(exist_ok=True)

    for letter,(x,top,w,h) in PANEL_BOXES.items():

        box=Bbox.from_bounds(x/25.4,(HEIGHT-top-h)/25.4,w/25.4,h/25.4)

        for suffix in ("svg","pdf","png"):

            kw={"metadata":metadata} if suffix=="svg" else {}

            fig.savefig(panels/f"{letter}.{suffix}",bbox_inches=box,pad_inches=0,dpi=600,facecolor="white",**kw)

    text=stem.with_suffix(".svg").read_text(encoding="utf-8")

    checks={"source_hashes_exact":True,"level_comparison_hashes_exact":True,"first30_hashes_exact":True,

        "distribution_source_hashes_exact":True,"fifteen_panels_three_rows":len(CROSSWALK)==15 and set(v[3] for v in CROSSWALK)=={1,2,3},

        "distribution_248_observations":evidence["a"]["observations"]==248 and evidence["a"]["null_values"]==2480000,

        "distribution_labels_clear":not distribution_overlaps,

        "level_population_vector_axis_names":axes["a_population_vector_distribution"].get_xlabel()=="Level PV correlation" and axes["b_population_vector_null"].get_xlabel()=="Mean level PV\ncorrelation",

        "no_geometric_eight_visible":not any(term in text for term in ["Geometric","geometric","Tree-depth","tree-depth","tree_depth7"]),

        "population_vector_heatmap_moved_out":"Earlier level" not in text and "Later level" not in text,

        "population_shuffle_10000_draws":evidence["b"]["null_draws"]==10000,"population_shuffle_hashes_exact":True,

        "mean_null_10000_draws":evidence["c"]["null_draws"]==10000,"panel_c_data_label_clears_line":bool(c_clearance_pt>=3),

        "decoder_36_pairs_14_animals":evidence["d"]["pairs"]==36 and evidence["d"]["animals"]==14,

        "level_comparison_36_pairs":evidence["h"]["pairs"]==36,

        "middle_row_order":[v[1] for v in CROSSWALK if v[3]==2]==["g","h","f","i"],

        "bottom_row_order":[v[1] for v in CROSSWALK if v[3]==3]==["j","k","l","m","n"],

        "first30_rows_and_animals":evidence["n"]["rows"]==59 and evidence["n"]["animals"]==22,

        "top_frames_aligned":all(abs(geometry[n]["bounds_mm"][1]-10)<1e-9 and abs(geometry[n]["bounds_mm"][3]-26)<1e-9 for n in ["a_population_vector_distribution","b_population_vector_null","c_mean_null","d_decoding","e_transition","f_shift"]),

        "middle_quantitative_frames_aligned":all(abs(geometry[n]["bounds_mm"][1]-51)<1e-9 and abs(geometry[n]["bounds_mm"][3]-26)<1e-9 for n in ["h_level_controls","i_joint_cloud","j_pair","j_null"]),

        "maze_heatmaps_match_middle_plot_height":all(abs(geometry[n]["bounds_mm"][1]-51)<1e-9 and abs(geometry[n]["bounds_mm"][2]-26)<1e-9 and abs(geometry[n]["bounds_mm"][3]-26)<1e-9 for n in ["g_horizontal","g_vertical"]),

        "bottom_frames_aligned":all(abs(geometry[n]["bounds_mm"][1]-91)<1e-9 and abs(geometry[n]["bounds_mm"][3]-31)<1e-9 for n in ["k_gaps","l_consecutive","m_learning","n_first30","o_cycles"]),

        "ordinary_axis_labels_same_size":all(ax.xaxis.label.get_fontsize()==FONT and ax.yaxis.label.get_fontsize()==FONT for ax in axes.values()),

        "long_gap_zero_line_removed":not any(np.all(np.asarray(line.get_ydata())==0) for line in axes["o_cycles"].lines),

        "no_text_overflow":not text_overflow}

    write_json(output/"VALIDATION.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"overflow_candidates":text_overflow,"dimensions_mm":[WIDTH,HEIGHT],"source_inputs":len(input_audit),"quantitative_source":"frozen numerical tables; no accepted artwork read"})

    plt.close(fig)

    assert all(checks.values()), checks

    print(json.dumps({"status":"PASS","output":str(output),"checks":len(checks),"overflow_candidates":text_overflow}))





if __name__ == "__main__":

    parser=argparse.ArgumentParser(description=__doc__)

    source_dir = Path(__file__).resolve().parent

    default = (source_dir if (source_dir / "SOURCE_INPUTS.json").exists()

               else source_dir.parent / "output/figures/ed9_distribution_reflow_2026-09-26")

    parser.add_argument("--input-dir",type=Path,default=default)

    parser.add_argument("--output-dir",type=Path,default=default)

    args=parser.parse_args()

    build(args.input_dir.resolve(),args.output_dir.resolve())

