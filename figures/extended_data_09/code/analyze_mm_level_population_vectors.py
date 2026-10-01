"""Recompute MM049 depth population vectors from the current 36-pair authority.

The inputs are the unnormalized C-derived maps underlying the current Main
Figure 5 heatmap. This changes the correlation axis from depths within a neuron
to registered neurons at a depth. Maintained figures and pointers are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import findfont
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "docs/main_figs_v3_source_render_2026-06-24/figure5_no_manual_mask_2026-08-07"
FIG2 = ROOT / "docs/sorted_data_full_figure_rerun_2026-06-11/figure2_distribution_shuffle_corrected_842_no_presort_exclusions_2026-08-03"
STYLE_SOURCE = ROOT / "docs/figure_version_records/main_figure2_2026-09-08_round11_count_titles_geometry_preserved/source_code/render_main_figs_v3_publication.py"
DEFAULT_OUT = ROOT / "output/analysis/mm_proposed_panels_2026-09-24/level_pv"
PV = "#E0954E"
GRAY = "#737373"
DT = 0.04
UNREWARDED_SOURCE_SESSIONS = {"837day1", "838day1", "839day1", "842day1", "862day1"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def row_unit(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.full_like(centered, np.nan), where=norms > 1e-12)


def col_corr(a, b):
    return np.sum(row_unit(a.T) * row_unit(b.T), axis=1)


def clustered(values, animals):
    """Intercept-only OLS with CR1 and a two-sided t(G-1) reference."""
    y = np.asarray(values, dtype=float)
    animals = np.asarray(animals).astype(str)
    keep = np.isfinite(y)
    y, animals = y[keep], animals[keep]
    groups = np.unique(animals)
    n, g = len(y), len(groups)
    estimate = float(y.mean())
    scores = np.array([(y[animals == a] - estimate).sum() for a in groups])
    se = float(np.sqrt(g / (g - 1) * np.sum(scores**2) / n**2))
    t = estimate / se if se > 0 else np.nan
    q = float(stats.t.ppf(0.975, g - 1))
    return dict(estimate=estimate, se=se, ci_low=estimate-q*se, ci_high=estimate+q*se,
                t=t, df=g-1, p_two_sided=float(2*stats.t.sf(abs(t),g-1)), n_pairs=n, n_animals=g)


def exact_cluster_signflip(y, animals):
    """Small-cluster calibration: synchronized signs on animal score sums."""
    y, animals = np.asarray(y, float), np.asarray(animals).astype(str)
    good = np.isfinite(y)
    y, animals = y[good], animals[good]
    groups = np.unique(animals)
    sums = np.array([y[animals == a].sum() for a in groups])
    signs = 2*((np.arange(2**len(groups))[:, None] >> np.arange(len(groups))) & 1)-1
    null = signs @ sums / len(y)
    return float(np.mean(np.abs(null) >= abs(y.mean())-1e-12))


def style():
    findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial", "font.size":6.3, "axes.labelsize":7,
        "axes.titlesize":7.2, "xtick.labelsize":6.3, "ytick.labelsize":6.3,
        "legend.fontsize":5.7, "axes.linewidth":.65, "svg.fonttype":"none",
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white",
        "mathtext.fontset":"custom", "mathtext.rm":"Arial", "mathtext.it":"Arial:italic",
        "mathtext.bf":"Arial:bold"})


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.55, length=1.8, pad=2)


def save_panel(fig, out, stem):
    for suffix in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out/f"{stem}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def raw_validation(out, sources):
    """Independent bounded raw-C reconstruction, including corrected 842."""
    import h5py
    from ca_analysis.io import _extract_info
    from ca_analysis.maze import NewMaze
    from ca_analysis.spatial import assign_levels, get_level_list
    maze = NewMaze(6)
    levels = get_level_list("tree_depth7", maze=maze)
    rows, occupancy, inputs = [], [], []
    for key in ("418RRday1->418-3day1", "842day1->842-3day1"):
        source = sources.loc[key]
        with (ROOT/source.selected_pkl_path).open("rb") as f:
            saved = pickle.load(f)
        for side, session in (("train", saved["src_session"]), ("test", saved["dst_session"])):
            beh_path = Path("D:/combineddata/selected_data")/f"{session}_selected_data.csv"
            df = pd.read_csv(beh_path, usecols=["frame", "mass_bx", "mass_by"])
            animal, experience, day = _extract_info(session)
            nfolder = animal if not experience else f"{animal}-{experience}"
            neural_path = Path("D:/neuralData_sort")/nfolder/f"final_day{day}.mat"
            ids = saved[f"{side}_ids"]
            frames = df.frame.to_numpy(int)
            valid = np.ones(len(df), bool)
            sample_c = np.empty((len(df), len(ids)), dtype=float)
            with h5py.File(neural_path, "r") as h:
                c = h["C"]
                for start in range(0, int(frames.max())+1, 8192):
                    stop = min(start+8192, c.shape[0])
                    take = (frames >= start)&(frames < stop)
                    if not take.any():
                        continue
                    block = c[start:stop]
                    selected = block[frames[take]-start]
                    valid[take] = ~np.isnan(selected).any(axis=1)
                    sample_c[take] = selected[:, ids]
            kept = df.loc[valid].reset_index(drop=True)
            depth = np.array(assign_levels(kept, maze.ce, levels), dtype=float)
            c_selected = sample_c[valid]
            reconstructed = np.empty((7,len(ids)))
            for d in range(7):
                take = depth == d
                n = int(take.sum())
                reconstructed[d] = np.maximum(c_selected[take].sum(axis=0)/(n*DT+1e-12),1e-12) if n else 1e-12
                occupancy.append(dict(pair_key=key,session=session,side=side,depth=d,level=d+1,
                    retained_samples=n,retained_exposure_s=n*DT))
            original = saved[f"{side}_map"]
            max_abs = float(np.max(np.abs(original-reconstructed)))
            # The maintained tuner adds its epsilon pseudocount before division.
            np.testing.assert_allclose(original,reconstructed,rtol=2e-7,atol=2e-9)
            rows.append(dict(pair_key=key,session=session,side=side,n_neurons=len(ids),
                selected_rows=len(df),retained_rows=len(kept),nan_selected_rows_removed=int((~valid).sum()),
                unassigned_rows=int(np.count_nonzero(~np.isfinite(depth))),
                max_abs_map_error=max_abs, raw_C_reconstruction="PASS",
                max_absolute_frame_gap=int(np.diff(kept.frame).max()),
                n_discontinuous_frame_gaps=int(np.sum(np.diff(kept.frame)>1)),
                timebase="absolute frame index",signal="C",neural_source="sorted"))
            for p in (beh_path, neural_path):
                inputs.append(dict(path=str(p),sha256=sha256(p),role="bounded raw reconstruction"))
            print(f"Raw validation {session}: max_abs={max_abs:.3g}",flush=True)
    pd.DataFrame(rows).to_csv(out/"raw_C_reconstruction_audit.csv",index=False)
    pd.DataFrame(occupancy).to_csv(out/"raw_validation_depth_exposure.csv",index=False)
    return inputs


def compute(out, repeats, seed):
    authority_path = AUTHORITY/"displayed_panel_b_heatmap_pair_source_audit.csv"
    source = pd.read_csv(authority_path).set_index("pair_key",drop=False)
    fig2 = pd.read_csv(FIG2/"pair_support_audit.csv").set_index("pair_key")
    pair_inputs = pd.read_csv(FIG2/"pair_input_audit.csv").set_index("pair_key")
    assert len(source) == 36 and source.animal_id.nunique() == 14
    assert set(source.index) == set(fig2.index)
    assert source.n_neurons.sum() == 16901
    rng = np.random.default_rng(seed)
    matrices, pair_rows, level_rows, neuron_frames, audits, input_hashes = [],[],[],[],[],[]
    intermediates = out/"intermediates"
    intermediates.mkdir(exist_ok=True)
    for order, row in enumerate(source.itertuples(index=False)):
        key = row.pair_key
        src_path = ROOT/row.selected_pkl_path
        source_hash = sha256(src_path)
        assert source_hash == row.selected_pkl_sha256
        with src_path.open("rb") as f:
            x = pickle.load(f)
        cache_path = Path(fig2.loc[key,"raw_pair_cache"])
        with cache_path.open("rb") as f:
            f2 = pickle.load(f)["result"]
        a,b = np.asarray(x["train_map"],float), np.asarray(x["test_map"],float)
        mask = np.asarray(x["train_mask"],bool)&np.asarray(x["test_mask"],bool)
        assert x["level_definition"] == "tree_depth7" and a.shape == b.shape == (7,row.n_neurons)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.array_equal(x["level_categories"],np.arange(7))
        assert np.array_equal(x["train_ids"],f2["train_ids"])
        assert np.array_equal(x["test_ids"],f2["test_ids"])
        gate = x["place_gate_summary"]
        assert gate["min_spikes_per_min"] == .1 and gate["rate_source"] == "spike"
        assert gate["drop_if_below"] == "either"
        saved_corr = np.asarray(x["similarity"])
        recomputed_corr = col_corr(a[mask],b[mask])
        np.testing.assert_allclose(saved_corr,recomputed_corr,atol=2e-10,equal_nan=True)
        u,v = row_unit(a),row_unit(b)
        m = u@v.T
        matrix_mask = mask[:,None]&mask[None,:]&np.isfinite(m)
        m[~matrix_mask] = np.nan
        diag = np.diag(m)
        off = m.copy()
        np.fill_diagonal(off,np.nan)
        # Average outgoing and incoming nonmatching depths at each target depth.
        different = (np.nansum(off,axis=0)+np.nansum(off,axis=1))/(2*(mask.sum()-1))
        different[~mask] = np.nan
        nulls = np.empty((repeats,7),float)
        for r in range(repeats):
            perm = rng.permutation(a.shape[1])
            nulls[r] = np.sum(u*v[:,perm],axis=1)
        nulls[:,~mask] = np.nan
        shuffle = np.full(7,np.nan)
        shuffle[mask] = np.mean(nulls[:,mask],axis=0)
        common = dict(pair_key=key,animal_id=str(row.animal_id),src_session=x["src_session"],
            dst_session=x["dst_session"],experience=pair_inputs.loc[key,"experience"],
            transition=pair_inputs.loc[key,"transition"],n_neurons=a.shape[1],n_depths=int(mask.sum()))
        common.update(source_reward_condition="unrewarded" if x["src_session"] in UNREWARDED_SOURCE_SESSIONS else "rewarded",
                      target_reward_condition="rewarded")
        pair_rows.append({**common,"same_depth":np.nanmean(diag),"different_depth":np.nanmean(off),
            "neuron_id_shuffle":np.nanmean(shuffle),"depth_specificity":np.nanmean(diag)-np.nanmean(off),
            "identity_preservation":np.nanmean(diag)-np.nanmean(shuffle)})
        for d in range(7):
            level_rows.append({**common,"depth":d,"level":d+1,"supported":bool(mask[d]),
                "train_mask":bool(x["train_mask"][d]),"test_mask":bool(x["test_mask"][d]),
                "same_depth":diag[d],"different_depth":different[d],"neuron_id_shuffle":shuffle[d],
                "depth_specificity":diag[d]-different[d],"identity_preservation":diag[d]-shuffle[d]})
        for d1 in range(7):
            for d2 in range(7):
                matrices.append({**common,"train_depth":d1,"test_depth":d2,"train_level":d1+1,
                    "test_level":d2+1,"supported":bool(matrix_mask[d1,d2]),"pearson_r":m[d1,d2]})
        neurons = pd.DataFrame({"pair_key":key,"animal_id":str(row.animal_id),
            "train_id":x["train_ids"],"test_id":x["test_ids"],"source_level_map_r":saved_corr})
        for d in range(7):
            neurons[f"train_depth_{d}_C_per_s"] = a[d]
            neurons[f"test_depth_{d}_C_per_s"] = b[d]
        neuron_frames.append(neurons)
        np.savez_compressed(intermediates/f"{order+1:02d}_{x['src_session']}_to_{x['dst_session']}.npz",
            train_map=a,test_map=b,train_ids=x["train_ids"],test_ids=x["test_ids"],
            train_mask=x["train_mask"],test_mask=x["test_mask"],common_mask=mask,
            train_centered_unit_vectors=u,test_centered_unit_vectors=v,matrix=m,
            neuron_id_shuffle_by_depth=nulls.astype(np.float32))
        audits.append({**common,"source_pkl":str(src_path),"source_sha256":source_hash,
            "matching_source":x.get("matching_source"),"neuron_policy":x["neuron_policy"],
            "gate_n_initial":gate["n_pairs_initial"],"gate_n_removed":gate["n_pairs_dropped"],
            "gate_n_kept":gate["n_pairs_kept"],"fig2_neuron_ID_parity":"PASS",
            "map_r_reconstruction_max_abs":float(np.nanmax(np.abs(saved_corr-recomputed_corr))),
            "excluded_depths":json.dumps(np.flatnonzero(~mask).tolist()),
            "CIMap_path":pair_inputs.loc[key,"effective_cimap_path"],
            "CIMap_sha256":sha256(pair_inputs.loc[key,"effective_cimap_path"]),
            "corrected_842_override":pair_inputs.loc[key,"corrected_842_override"]})
        input_hashes.extend([dict(path=str(src_path),sha256=source_hash,role="current raw level-map authority"),
            dict(path=str(cache_path),sha256=sha256(cache_path),role="current Figure 2 neuron-ID parity")])
        print(f"{order+1}/36 {key}: same={pair_rows[-1]['same_depth']:.3f}, specificity={pair_rows[-1]['depth_specificity']:.3f}",flush=True)
    pair_df,levels_df,matrix_df = pd.DataFrame(pair_rows),pd.DataFrame(level_rows),pd.DataFrame(matrices)
    pd.DataFrame(audits).to_csv(out/"source_support_and_registration_audit.csv",index=False)
    pair_df.to_csv(out/"pair_summary.csv",index=False)
    levels_df.to_csv(out/"pair_by_depth.csv",index=False)
    matrix_df.to_csv(out/"pair_depth_cross_similarity_matrices.csv",index=False)
    pd.concat(neuron_frames,ignore_index=True).to_csv(out/"registered_neuron_depth_maps.csv",index=False)
    source.reset_index(drop=True).to_csv(out/"frozen_source_selection.csv",index=False)
    pd.read_csv(FIG2/"session_input_audit.csv").to_csv(out/"inherited_session_alignment_audit.csv",index=False)
    tests = []
    for metric in ("identity_preservation","depth_specificity"):
        tests.append(dict(contrast=metric,**clustered(pair_df[metric],pair_df.animal_id),
            exact_animal_score_signflip_p=exact_cluster_signflip(pair_df[metric],pair_df.animal_id)))
    tests_df = pd.DataFrame(tests)
    tests_df["p_BH2"] = multipletests(tests_df.p_two_sided,method="fdr_bh")[1]
    tests_df["exact_signflip_p_BH2"] = multipletests(tests_df.exact_animal_score_signflip_p,method="fdr_bh")[1]
    tests_df.to_csv(out/"global_clustered_inference.csv",index=False)
    summary=[]
    for d, frame in levels_df.groupby("depth"):
        for metric in ("same_depth","different_depth","neuron_id_shuffle","depth_specificity","identity_preservation"):
            summary.append(dict(depth=d,level=d+1,metric=metric,**clustered(frame[metric],frame.animal_id)))
    summary_df = pd.DataFrame(summary)
    summary_df["p_BH7_within_contrast"] = np.nan
    for metric in ("depth_specificity","identity_preservation"):
        sel = summary_df.metric.eq(metric)
        summary_df.loc[sel,"p_BH7_within_contrast"] = multipletests(summary_df.loc[sel,"p_two_sided"],method="fdr_bh")[1]
    summary_df.to_csv(out/"depth_curve_clustered_summary.csv",index=False)
    animal = pair_df.groupby("animal_id",as_index=False)[["same_depth","different_depth","neuron_id_shuffle","depth_specificity","identity_preservation"]].mean()
    animal.to_csv(out/"animal_mean_weighting_sensitivity.csv",index=False)
    pd.DataFrame([dict(contrast=k,n_animals=len(animal),mean=float(animal[k].mean()),
        sem=float(stats.sem(animal[k])),p_two_sided=float(stats.ttest_1samp(animal[k],0).pvalue))
        for k in ("identity_preservation","depth_specificity")]).to_csv(out/"animal_mean_sensitivity_tests.csv",index=False)
    support_sensitivity=[]
    for label,frame in (("all36_native_common_depths",matrix_df),
            ("all36_fixed_levels1to6",matrix_df.loc[matrix_df.train_depth.lt(6)&matrix_df.test_depth.lt(6)]),
            ("complete7levels_32pairs",matrix_df.loc[matrix_df.pair_key.isin(pair_df.loc[pair_df.n_depths.eq(7),"pair_key"])]),
            ("rewarded_source_only_31pairs",matrix_df.loc[matrix_df.source_reward_condition.eq("rewarded")])):
        contrast_rows=[]
        for key,g in frame.groupby("pair_key"):
            same=g.loc[g.train_depth.eq(g.test_depth),"pearson_r"].mean()
            diff=g.loc[g.train_depth.ne(g.test_depth),"pearson_r"].mean()
            contrast_rows.append(dict(pair_key=key,animal_id=g.animal_id.iloc[0],same_depth=same,
                different_depth=diff,depth_specificity=same-diff))
        table=pd.DataFrame(contrast_rows)
        table.to_csv(out/f"support_sensitivity_{label}.csv",index=False)
        support_sensitivity.append(dict(support_policy=label,**clustered(table.depth_specificity,table.animal_id)))
    pd.DataFrame(support_sensitivity).to_csv(out/"support_sensitivity_clustered_tests.csv",index=False)
    for p in (authority_path,FIG2/"pair_support_audit.csv",FIG2/"pair_input_audit.csv",FIG2/"session_input_audit.csv",
              AUTHORITY/"SOURCE_MANIFEST.json",STYLE_SOURCE,Path(__file__),ROOT/"scripts/export_long_gap_level_tuning_maps.py",
              ROOT/"src/ca_analysis/similarity.py",ROOT/"src/ca_analysis/spatial.py"):
        input_hashes.append(dict(path=str(p),sha256=sha256(p),role="input/provenance/code"))
    return pair_df,levels_df,matrix_df,summary_df,tests_df,source,input_hashes


def render(out,pairs,levels,matrices,summary,tests):
    style()
    matrix = matrices.groupby(["train_depth","test_depth"]).pearson_r.mean().unstack().to_numpy()
    support = matrices.groupby(["train_depth","test_depth"]).pearson_r.count().unstack().to_numpy()
    pd.DataFrame(matrix,index=range(7),columns=range(7)).to_csv(out/"mean_cross_depth_matrix.csv",index_label="train_depth")
    pd.DataFrame(support,index=range(7),columns=range(7)).to_csv(out/"matrix_pair_support.csv",index_label="train_depth")
    cmap = LinearSegmentedColormap.from_list("pv_orange",["#ffffff",PV,"#8f4b18"])

    def heat(ax, fig):
        edges=np.arange(8)-.5
        im=ax.pcolormesh(edges,edges,matrix,cmap=cmap,vmin=0,vmax=.65,
                         shading="flat",rasterized=False,linewidth=0)
        ax.set(xlim=(-.5,6.5),ylim=(6.5,-.5),aspect="equal")
        ax.set_xticks(range(7),range(1,8));ax.set_yticks(range(7),range(1,8))
        ax.set_xlabel("Later level");ax.set_ylabel("Earlier level")
        ax.set_title("Across-depth population similarity",pad=7)
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.06)
        cb.solids.set_rasterized(False)
        cb.set_label("Population-vector correlation",labelpad=5)
        cb.ax.tick_params(width=.55,length=1.8,labelsize=6.3)
        for d in range(7):
            ax.add_patch(plt.Rectangle((d-.5,d-.5),1,1,fill=False,edgecolor="#333333",lw=.55))
        ax.tick_params(width=.55,length=1.8,pad=2)

    def curve(ax):
        for metric,label,color,ls in (("same_depth","Same level",PV,"-"),
                ("different_depth","Different level",GRAY,"--"),("neuron_id_shuffle","Neuron-ID shuffle","#b7b7b7",":")):
            g=summary.loc[summary.metric.eq(metric)].sort_values("depth")
            ax.plot(g.level,g.estimate,color=color,lw=1.1,ls=ls,label=label)
            ax.fill_between(g.level,g.ci_low,g.ci_high,color=color,alpha=.10,lw=0)
        ax.set(xticks=range(1,8),xlabel="Maze level",ylabel="Population-vector correlation",xlim=(.8,7.2),ylim=(-.035,.70))
        ax.legend(loc="upper left",frameon=False,handlelength=1.8)
        ax.set_title("Population correlation at each level",pad=7)
        axis_style(ax)

    def paired(ax,animal=False):
        data = pairs.groupby("animal_id",as_index=False)[["same_depth","different_depth","neuron_id_shuffle"]].mean() if animal else pairs
        cols=["same_depth","different_depth","neuron_id_shuffle"]
        for _,r in data.iterrows():
            ax.plot(range(3),r[cols].to_numpy(float),color="#b8b8b8",alpha=.6,lw=.4,zorder=1)
        for j,(col,c) in enumerate(zip(cols,[PV,GRAY,"#b7b7b7"])):
            ax.scatter(np.full(len(data),j),data[col],s=5,facecolor=c,edgecolor="none",alpha=.65,zorder=2)
            res = clustered(pairs[col],pairs.animal_id) if not animal else None
            mean,err=(res["estimate"],stats.t.ppf(.975,res["df"])*res["se"]) if res else (data[col].mean(),stats.t.ppf(.975,len(data)-1)*stats.sem(data[col]))
            ax.errorbar(j+.13,mean,yerr=err,fmt="o",ms=2.6,color="black",lw=.7,capsize=2,zorder=3)
        ax.set(xticks=range(3),xticklabels=["Same\nlevel","Different\nlevel","Neuron-ID\nshuffle"],
            ylabel="Population-vector correlation",xlim=(-.35,2.4),ylim=(-.13,.91))
        ax.set_title("Depth-specific preservation" if not animal else "Equal-animal weighting",pad=7)
        p=float(tests.loc[tests.contrast.eq("depth_specificity"),"p_BH2"].iloc[0])
        exponent=int(np.floor(np.log10(p)))
        coefficient=p/10**exponent
        p_text=rf"$p = {coefficient:.1f} \times 10^{{{exponent}}}$"
        ax.text(.02,.98,f"36 pairs, 14 mice\nSame − different:\n{p_text}" if not animal else "14 mouse means",
            transform=ax.transAxes,va="top",fontsize=6.3)
        axis_style(ax)

    for name,func in (("level_pv_a_cross_depth_matrix",heat),("level_pv_b_by_depth",curve),("level_pv_c_pair_comparison",paired)):
        fig,ax=plt.subplots(figsize=(2.35,2.15))
        fig.subplots_adjust(left=.22,right=.89,bottom=.22,top=.84)
        if func is heat:
            fig.subplots_adjust(left=.20,right=.74,bottom=.21,top=.80)
        if func is paired:
            fig.subplots_adjust(right=.97)
        func(ax,fig) if func is heat else func(ax)
        save_panel(fig,out,name)
    fig,axs=plt.subplots(1,3,figsize=(180/25.4,2.25),gridspec_kw={"width_ratios":[1.08,1,1]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.80,wspace=.55)
    heat(axs[0],fig);curve(axs[1]);paired(axs[2])
    for letter,ax in zip("abc",axs):
        ax.text(-.22,1.25,letter,transform=ax.transAxes,fontweight="bold",fontsize=11.5,va="top")
    save_panel(fig,out,"level_population_vectors_publication")
    fig,ax=plt.subplots(figsize=(2.35,2.15));fig.subplots_adjust(left=.22,right=.94,bottom=.24,top=.84)
    paired(ax,animal=True);save_panel(fig,out,"level_pv_sensitivity_animal_means")


def report(out,pairs,levels,tests,repeats,raw_done):
    values=pairs[["same_depth","different_depth","neuron_id_shuffle","depth_specificity"]].mean()
    test_table = "| Contrast | Estimate | 95% CI | p (BH2) | Exact animal-score p |\n|---|---:|---|---:|---:|\n"
    for r in tests.itertuples():
        test_table += f"| {r.contrast} | {r.estimate:.6f} | [{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p_BH2:.6g} | {r.exact_animal_score_signflip_p:.6g} |\n"
    text=f"""# Population vectors for maze levels (MM049)

Across 36 approximately three-month Day-1 recording pairs from 14 mice, mean same-level population-vector correlation was {values.same_depth:.6f}, versus {values.different_depth:.6f} for nonmatching levels. Their mean difference was {values.depth_specificity:.6f}. This is a population-vector correlation across neurons at one depth, distinct from the existing per-neuron correlation across depths. The complete current cohort contributes 16,901 registered neuron-pair observations; these are repeated observations, not 16,901 independent biological replicates.

## Definition and support

The representation is the current seven tree depths 0–6 (display levels 1–7), not the older geometric eight-region partition. The cohort exactly matches current Figure 2 and includes five initially unrewarded source sessions (837day1, 838day1, 839day1, 842day1 and 862day1) paired with their rewarded returns; the other 31 source sessions and all target sessions are rewarded, with endpoint node 94. Reward status is exported per pair, and a 31-pair rewarded-source-only sensitivity is provided. No event-aligned or within-session learning claim is made. The current Main Figure 5 raw level maps are reused before their display L2 normalization. In each session and depth, the map is the summed denoised calcium C divided by retained depth exposure (0.04 s per retained sample). No neuron-wise normalization, field-peak selection, smoothing, or Fisher transform is applied to the population readout. Row centering and unit length implement Pearson correlation across registered neurons.

The inherited sorted CIMaps and the whole-session S activity gate (at least 0.1/min in both sessions, no manual exclusions) are exact neuron-ID matches to current Figure 2 for all 36 pairs. The corrected 842 0-to-3-month CIMap is explicitly checked. Source masks require positive occupancy in each session; this source-level rule is inherited, not represented as a new five-second cutoff. Four pairs lack depth 6 in one session; their common six levels are used. Thus 36 pairs support each level 1–6 and 32 support level 7. A saved unsupported depth is never replaced by zero correlation.

For each pair, compute a 7×7 cross-session Pearson matrix using a fixed matched population. Both matrix axes use the common supported-depth mask. The same-level readout averages its diagonal; the nonmatching readout averages its off-diagonal entries. Per-level nonmatching controls average correlations to all other common levels in both temporal directions. This controls for persistent overall neuron activity ordering; the difference estimates depth specificity beyond that ordering. Neuron-ID controls apply {repeats:,} independent within-pair test-neuron permutations, using one permutation shared over depths in each repeat. Their technical mean is one control value per pair. The identity-shuffle control alone cannot establish depth specificity.

## Inference and panel key

All primary means weight session pairs equally. Error bands and bars are 95% confidence intervals from an intercept-only model with CR1 animal-clustered covariance and Student t(G−1), retaining repeated pairs. Two global contrasts (same level minus neuron-ID control; same level minus different level) form a BH2 family. Exact animal-score sign flipping across all 16,384 sign patterns is a calibration sensitivity, and equal-animal weighting is exported separately. Per-level comparisons are secondary BH7 families within each contrast. Neurons, depths and shuffle repeats are never treated as independent biological replicates.

- a: Mean 7×7 cross-depth similarity, equal weight to available pairs per matrix cell. Boxed diagonal marks matching depths; cell-specific support is exported.
- b: Same-level, nonmatching-level and neuron-ID control across maze levels; shading shows animal-clustered 95% confidence intervals.
- c: Each joined triplet is one recording pair. Black points and error bars show pair-weighted mean and animal-clustered 95% CI. The displayed p is BH2-adjusted.
- Supplement: one mean per mouse, with ordinary mouse-level 95% CI, supplied only as a weighting sensitivity.

The source-data bundle also tests the specificity contrast on the fixed first six levels in all 36 pairs, on all seven levels in the 32 complete pairs, and in the 31 pairs with rewarded source sessions. These are support/reward-condition sensitivities, not additional confirmatory tests.

{test_table}

## Alignment, conversion and reproducibility

No new cross-modal merge is performed. Saved source maps used absolute neural frame indices from the speed-thresholded, potentially discontinuous selected_df. C-NaN rows were removed before map fitting; exposure uses retained sample counts rather than endpoint differences. The inherited session-level alignment audit, including frame-gap counts and excluded-row counts, is copied as `inherited_session_alignment_audit.csv`. Each saved cell-ID pair is checked against the current Figure 2 raw cache, every source hash is checked against the current heatmap source audit, and every per-neuron level-map correlation is independently recalculated before changing correlation axes.

`registered_neuron_depth_maps.csv` exposes all raw map entries; `intermediates/*.npz` additionally contains centered/unit population vectors, the exact masks, neuron IDs, correlation matrices and every shuffled per-depth correlation. `pair_by_depth.csv`, `pair_depth_cross_similarity_matrices.csv` and `pair_summary.csv` expose all aggregations. The transformation is (7 depths × N registered neurons) → (7 × 7 cross-session matrix) → depth/pair summaries; the per-neuron source r is reproduced only as validation and is not the new outcome.

Independent raw-C reconstruction of 418 and corrected842 sessions: {'completed; see raw_C_reconstruction_audit.csv and raw_validation_depth_exposure.csv' if raw_done else 'not requested on this invocation'}. The complete raw map extraction is inherited from the hashed manuscript authority; this script recomputes all new PV coefficients, nulls, summaries and statistics from those raw maps.

Rebuild: `python scripts/analyze_mm_level_population_vectors.py --validate-raw`.

Exports match the measured physical Arial sizes in the assembled manuscript (7-pt axes, 6.3-pt ticks, 7.2-pt titles, 5.7-pt legends, 11.5-pt bold panel letters; 180-mm composite), actual Figure 2 PV orange, editable SVG/PDF text, and 600-dpi PNG/TIFF. Current manuscript files, maintained figure pointers and captions are unchanged.

## Limits

This asks whether the population pattern recurs over a long gap; it does not establish that a neural state persisted continuously during the gap, that it is causally necessary, or that depth is abstracted from all sensory/motor correlates. Same-level versus nonmatching-level controls overall rate ordering but not different spatial occupancy within each depth. This analysis does not add a Day-1/Day-2 anchor because the current same-stage exports contain scalar summaries rather than comparable saved raw depth vectors; attaching those older level-map scalars would change the estimand.
"""
    (out/"README.md").write_text(text,encoding="utf-8")


def audit_exports(out):
    import fitz
    from PIL import Image
    rows=[]
    for path in sorted(out.glob("*.pdf")):
        with fitz.open(path) as doc:
            page=doc[0]
            spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b
                   for line in b["lines"] for s in line["spans"]]
            clipped=[s["text"] for s in spans if s["bbox"][0]<0 or s["bbox"][1]<0
                     or s["bbox"][2]>page.rect.width+.5 or s["bbox"][3]>page.rect.height+.5]
            assert not clipped, (path,clipped)
            assert spans, f"PDF lacks editable text: {path}"
            raster_images=len(page.get_images(full=True))
            assert raster_images==0, f"Raster image found in vector PDF: {path}"
            with Image.open(path.with_suffix(".png")) as image:
                dpi=image.info.get("dpi",(0,0))
                assert abs(dpi[0]-600)<.02
                rows.append(dict(file=path.name,width_mm=page.rect.width/72*25.4,
                    height_mm=page.rect.height/72*25.4,pdf_text_spans=len(spans),
                    pdf_fonts=sorted(set(s["font"] for s in spans)),
                    text_sizes_pt=sorted(set(round(s["size"],2) for s in spans)),
                    clipped_text_spans=clipped,pdf_raster_image_count=raster_images,
                    raster_width_px=image.width,raster_height_px=image.height,
                    raster_dpi=dpi,status="PASS"))
    write_json(out/"EXPORT_QA.json",dict(panels=rows,automated_checks="PDF text bounds, embedded editable text, zero raster images in vector PDF, physical dimensions, PNG resolution",
        interpretation="Automated export checks complement direct visual inspection; they do not establish scientific validity."))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--repeats",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20260924)
    parser.add_argument("--validate-raw",action="store_true")
    parser.add_argument("--reuse-raw-validation",action="store_true",help="Verify and retain the existing bounded raw-C validation rather than reloading C arrays.")
    parser.add_argument("--render-only",action="store_true",help="Restyle verified saved numerical outputs without rerunning the analysis.")
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        old_manifest=json.loads((out/"MANIFEST.json").read_text(encoding="utf-8"))
        for record in old_manifest["output_hashes"]:
            p=out/record["path"]
            if p.suffix in (".csv",".npz"):
                assert sha256(p)==record["sha256"], f"Saved numerical output changed: {p}"
        pairs=pd.read_csv(out/"pair_summary.csv")
        levels=pd.read_csv(out/"pair_by_depth.csv")
        matrices=pd.read_csv(out/"pair_depth_cross_similarity_matrices.csv")
        summary=pd.read_csv(out/"depth_curve_clustered_summary.csv")
        tests=pd.read_csv(out/"global_clustered_inference.csv")
        inputs=old_manifest["input_hashes"]
        for record in inputs:
            if Path(record["path"])==Path(__file__):
                record["sha256"]=sha256(__file__)
        args.seed=old_manifest["seed"]
        args.repeats=old_manifest["shuffle_repeats"]
    else:
        pairs,levels,matrices,summary,tests,sources,inputs=compute(out,args.repeats,args.seed)
        if args.validate_raw:
            inputs.extend(raw_validation(out,sources))
        elif args.reuse_raw_validation:
            old_manifest=json.loads((out/"MANIFEST.json").read_text(encoding="utf-8"))
            old_hashes={r["path"]:r["sha256"] for r in old_manifest["input_hashes"]}
            for record in inputs:
                if record["role"]=="current raw level-map authority":
                    assert record["sha256"]==old_hashes[record["path"]]
            for record in old_manifest["input_hashes"]:
                if record["role"]=="bounded raw reconstruction":
                    assert sha256(record["path"])==record["sha256"]
                    inputs.append(record)
            for record in old_manifest["output_hashes"]:
                if record["path"] in ("raw_C_reconstruction_audit.csv","raw_validation_depth_exposure.csv"):
                    assert sha256(out/record["path"])==record["sha256"]
    render(out,pairs,levels,matrices,summary,tests)
    audit_exports(out)
    raw_done=args.validate_raw or args.reuse_raw_validation or (args.render_only and (out/"raw_C_reconstruction_audit.csv").exists())
    report(out,pairs,levels,tests,args.repeats,raw_done)
    write_json(out/"MANIFEST.json",dict(analysis="MM049 tree-depth population vectors",seed=args.seed,
        shuffle_repeats=args.repeats,n_pairs=len(pairs),n_animals=int(pairs.animal_id.nunique()),
        n_registered_neuron_pair_observations=int(pairs.n_neurons.sum()),input_hashes=inputs,
        statistical_unit="session pair with animal-clustered CR1 / Student t(G-1)",
        global_correction_family="BH2 across identity preservation and depth specificity",
        output_hashes=[dict(path=str(p.relative_to(out)),sha256=sha256(p),bytes=p.stat().st_size)
            for p in sorted(out.rglob("*")) if p.is_file() and p.name!="MANIFEST.json"]))
    print(tests.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
