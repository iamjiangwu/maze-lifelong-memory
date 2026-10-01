"""Build the complete Figure 4 from compact, frozen numerical sources.

Use --freeze once inside the repository to create the dated source bundle.
Thereafter --bundle B --output-dir O rebuilds solely from B, without sessions,
registrations, the analysis folder, or the current figure registry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
import pandas as pd
from scipy import stats
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REL = "output/figures/figure4_native_bins_2026-09-29"
RECORD_NAME = "main_figure4_2026-09-07_round9_numeric_schematic"
RULE = "2026-09-27-uniform-port-grid-v4"
ORANGE, GRAY = "#e29957", "#C4C4C4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def array_hash(a):
    a = np.asarray(a)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def signature(fig):
    """Numerical artists, labels and geometry, excluding the colormap lookup."""
    result = []
    for ax in fig.axes:
        result.append(dict(
            bounds=np.round(ax.get_position().bounds, 12).tolist(),
            xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
            images=[dict(shape=list(im.get_array().shape), values=array_hash(im.get_array()),
                         mask=array_hash(np.ma.getmaskarray(im.get_array())), clim=list(im.get_clim())) for im in ax.images],
            lines=[dict(xy=array_hash(line.get_xydata()), color=str(line.get_color()),
                        linestyle=line.get_linestyle()) for line in ax.lines],
            patches=[dict(vertices=array_hash(p.get_path().vertices), bounds=list(p.get_extents().bounds)) for p in ax.patches],
            collections=[dict(offsets=array_hash(c.get_offsets()),
                              paths=[array_hash(p.vertices) for p in c.get_paths()]) for c in ax.collections],
            texts=[(t.get_text(), list(t.get_position()), t.get_fontsize()) for t in ax.texts],
            xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=ax.get_title()))
    return result


def svg(fig, *, dpi=None):
    out = io.BytesIO()
    fig.savefig(out, format="svg", transparent=True, facecolor="none", bbox_inches=None,
                dpi=dpi, metadata={"Date": None})
    return out.getvalue()


def native_schematic_bins(fig):
    """Keep the original 15x15 cells through the tilted SVG/PDF transform.

    The historical helper repeats each bin 8x8 before drawing the image. At
    low export DPI that sheet is flattened to a 101x17 raster. Recover the
    exact repeated values and use unsampled image export, preserving the
    existing affine transform, extent, normalization and selected neurons.
    """
    images = [im for im in fig.findobj(match=AxesImage)
              if im.get_array().shape == (120, 120)]
    assert len(images) == 5, "Expected five repeated 15x15 template maps"
    for im in images:
        tiled = np.asarray(im.get_array())
        native = tiled[::8, ::8].copy()
        assert np.array_equal(tiled, np.repeat(np.repeat(native, 8, axis=0), 8, axis=1))
        im.set_data(native)
        im.set_interpolation("none")
        im.set_resample(False)


def ci(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    m = a.mean()
    e = stats.t.ppf(.975, len(a)-1) * stats.sem(a)
    return m, m-e, m+e


def freeze(bundle, repo):
    route = repo / "output/analysis/route_density_vs_pv_explained_2026-09-24"
    vor = repo / "docs/figure_version_records"
    record = vor / RECORD_NAME
    bundle.mkdir(parents=True, exist_ok=True)
    for d in ["source_code", "plot_ready", "provenance"]:
        (bundle / d).mkdir(exist_ok=True)
    source_rows = []

    def copy(source, relative):
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_rows.append(dict(source=source.as_posix(), frozen=relative,
                                sha256=sha(source), bytes=source.stat().st_size))

    helper = load("figure4_freeze_helper", vor / "rebuild_current_main_figures.py")
    check = helper.check_artifact("main_figure_4")
    assert helper._pointer_records()["main_figure_4"].name == RECORD_NAME
    dump(bundle / "provenance/ACCEPTED_SOURCE_CHECK.json", check)
    helper._load_publication_renderer(record, "figure4_freeze_publication")
    # Use the current candidate's exact helpers but never execute its renderer.
    common = load("figure4_freeze_route_common", route / "route_common.py")
    from ca_analysis.linearized_pf_utils import peak_normalize, rowwise_corr
    from ca_analysis.figure_panels import _order_linearized_heatmaps_by_earlier_peak
    candidate = route / "candidate_panels.py"
    node = next(n for n in ast.parse(candidate.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "heat_data")
    ns = dict(np=np, SMD=route / "signal_maps_primary", fig4e_pairs=common.fig4e_pairs,
              full_cimap=common.full_cimap, rowwise_corr=rowwise_corr, peak_normalize=peak_normalize,
              _order_linearized_heatmaps_by_earlier_peak=_order_linearized_heatmaps_by_earlier_peak)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(candidate), "exec"), ns)
    maps, membership, comparisons, map_sources = {}, [], [], {}
    for direction, group, key in [("home_to_water", "home_to_reward", "h2r"),
                                  ("water_to_home", "reward_to_home", "r2h")]:
        aa, bb, rows = [], [], []
        for pair in common.fig4e_pairs(direction):
            sources = [route / "signal_maps_primary" / (pair[k] + ".npz") for k in ["id1", "id2"]]
            zz = [np.load(p, allow_pickle=False) for p in sources]
            for p, z in zip(sources, zz):
                info = json.loads(str(z["info"].item()))
                assert info["rule_version"] == RULE
                assert np.array_equal(z[key + "__canonical_edges_tiles"], np.arange(23))
                map_sources[p.as_posix()] = sha(p)
            a_all, b_all = [z[key + "__all__S_rate"].astype(float) for z in zz]
            ids = common.full_cimap(pair["id1"], pair["id2"], len(a_all), len(b_all))
            a, b = a_all[ids[:, 0]], b_all[ids[:, 1]]
            keep = (np.sum(~np.isnan(a), axis=1) >= 11) & (np.nanmax(np.where(np.isnan(a), -np.inf, a), axis=1) > 0)
            keep &= np.isfinite(rowwise_corr(np.nan_to_num(a), np.nan_to_num(b)))
            assert np.isfinite(a[keep]).all() and np.isfinite(b[keep]).all(), "Primary heatmaps must have complete spatial support"
            a, b = np.nan_to_num(peak_normalize(a[keep])), np.nan_to_num(peak_normalize(b[keep]))
            if key == "r2h":
                a, b = a[:, ::-1], b[:, ::-1]
            aa.append(a); bb.append(b)
            for n1, n2 in ids[keep]:
                rows.append(dict(group=group, id1=pair["id1"], id2=pair["id2"],
                                 source_neuron_zero_based=int(n1), target_neuron_zero_based=int(n2)))
            comparisons.append(dict(group=group, id1=pair["id1"], id2=pair["id2"], n_neurons=int(keep.sum())))
            for z in zz:
                z.close()
        a, b = np.vstack(aa), np.vstack(bb)
        order = np.argsort(np.nanargmax(a, axis=1))[::-1]
        maps[key + "_day_i"], maps[key + "_day_i90"] = a[order], b[order]
        expected = ns["heat_data"](direction)
        assert np.array_equal(a[order], expected[0]) and np.array_equal(b[order], expected[1])
        for display_row, index in enumerate(order):
            membership.append({**rows[index], "display_row_bottom_zero": display_row,
                               "day_i_peak_home_oriented_bin": int(np.argmax(a[index]))})
    np.savez_compressed(bundle / "plot_ready/panel_e_heatmaps.npz", **maps)
    pd.DataFrame(membership).to_csv(bundle / "plot_ready/panel_e_neuron_membership.csv", index=False)
    pd.DataFrame(comparisons).to_csv(bundle / "plot_ready/panel_e_pair_counts.csv", index=False)
    profiles = pd.read_csv(route / "signal_profiles_primary.csv", dtype={"animal": str})
    profiles = profiles[(profiles.signal == "S") & (profiles.norm == "normalized")]
    profiles.to_csv(bundle / "plot_ready/panel_e_pair_profiles.csv", index=False)
    animals = profiles.groupby(["group", "animal", "bin"], as_index=False)[["pv", "pv_sh"]].mean()
    animals.to_csv(bundle / "plot_ready/panel_e_animal_profiles.csv", index=False)
    numbers = pd.read_csv(route / "signal_comparison_primary_numbers.csv")
    numbers = numbers[(numbers.panel == "main") & (numbers.signal == "S") & (numbers.norm == "normalized")]
    numbers.to_csv(bundle / "plot_ready/panel_e_statistics.csv", index=False)
    curve = []
    for group, g in animals.groupby("group"):
        for b, v in g.groupby("bin"):
            mean, lo, hi = ci(v.pv)
            curve.append(dict(group=group, bin=int(b), mean=mean, lo=lo, hi=hi,
                              shuffle=v.pv_sh.mean(), n_mice=len(v)))
        x = g.pivot(index="animal", columns="bin", values="pv")
        contrast = x[21] - x.loc[:, 0:20].mean(axis=1)
        row = numbers[(numbers.group == group) & (numbers.item == "port minus elsewhere")].iloc[0]
        assert np.isclose(contrast.mean(), row["mean"], atol=1e-12)
        assert len(contrast) == int(row.n_mice)
    pd.DataFrame(curve).to_csv(bundle / "plot_ready/panel_e_curve.csv", index=False)
    for name in ["render_main_figs_v3_publication.py", "figure_panels.py", "render_figure4_day_source_comparison.py",
                 "render_main_figure4_panel_f_ed10_k166.py"]:
        copy(record / "source_code" / name, "reference_record/source_code/" + name)
    for name in ["panel_c_example_map_rows.csv", "panel_d_native_reference_rows.csv", "panel_f_matched_pair_means.csv", "figure4_day_source_statistics.csv"]:
        copy(record / "plot_ready" / name, "reference_record/plot_ready/" + name)
    for name in ["figure4b_neuron_id_shuffle_pair_summary.csv", "figure4b_neuron_id_shuffle_statistics.json"]:
        copy(record / "source_data/neuron_id_shuffle" / name, "reference_record/source_data/neuron_id_shuffle/" + name)
    for p in (record / "illustration_data").rglob("*"):
        if p.is_file():
            copy(p, "reference_record/illustration_data/" + p.relative_to(record / "illustration_data").as_posix())
    copy(vor / "rebuild_current_main_figures.py", "source_code/rebuild_current_main_figures.py")
    copy(vor / "SOURCE_REBUILD_MAIN_RUNTIME.zip", "source_code/SOURCE_REBUILD_MAIN_RUNTIME.zip")
    copy(Path(__file__).resolve(), "source_code/build_figure4_standardized_pv.py")
    copy(repo / "output/figures/figure4_palette_options_2026-09-25/PALETTE_SELECTION.json", "provenance/PALETTE_SELECTION.json")
    dump(bundle / "provenance/CONTROL_COLOR_SELECTION.json", dict(status="AUTHOR_SELECTED", color="#C4C4C4",
        description="Lighter gray controls", user_selection="User selected the lighter gray full-figure version",
        scope="Neuron-ID shuffle in b/e, matched spatial controls in c/d; orange data and cividis maps retained",
        boundaries="Identical white under-outline on the orange reward and gray control squares in c",
        significance="Panel f stars black"))
    for name in ["candidate_panels.py", "route_common.py", "signal_profiles_primary.csv", "signal_comparison_primary_numbers.csv"]:
        source_rows.append(dict(source=(route / name).as_posix(), sha256=sha(route / name), bytes=(route / name).stat().st_size))
    dump(bundle / "provenance/SOURCE_PROVENANCE.json", dict(rule_version=RULE, sources=source_rows, maps=map_sources,
         heatmap_parity="Exact array equality with current candidate_panels.heat_data for both directions",
         heatmap_support="All displayed selected maps are finite in all 22 spatial bins",
         display_scaling="Each neuron/session map divided by its own peak; colour saturation at 0.6",
         pv_scaling="Each neuron/session map mean-subtracted and divided by SD across 22 equally weighted positions",
         null="100 within-pair neuron-ID shuffles, from the saved primary analysis",
         aggregation="Recording pairs averaged within animal, then animals equally weighted; 95% t intervals",
         grid="22 equal 3.8 cm bins along 83.6 cm home-port path; bin 0 home, bin 21 port"))
    # Preserve the complete existing caption, replacing only its e passage.
    author = (route / "manuscript_input_frozen.tex").read_text(encoding="utf-8")
    start = author.index("\\textbf{The reward location anchors selective place stability.}")
    end = author.index("\n}", start)
    caption = author[start:end]
    old_start, old_end = caption.index("\\textbf{e,}"), caption.index("\\textbf{f,}")
    proposal = (route / "manuscript_replacement_proposal.md").read_text(encoding="utf-8")
    p = proposal.split("## 4. Figure 4 caption, panel e", 1)[1].split("**Proposed**", 1)[1]
    proposed = p.split("```latex", 1)[1].split("```", 1)[0].strip()
    proposed = proposed.replace("Line plots show the population vector correlation", "Line plots show standardized population vector (PV) correlation")
    proposed = proposed.replace("after each neuron's map was standardized", "after subtracting each neuron's mean across route bins and dividing by its standard deviation, separately in each session")
    caption = caption[:old_start] + proposed + " \\\\\n" + caption[old_end:]
    caption = caption.replace("gray outlined bins are matched controls", "light gray outlined bins are matched controls")
    (bundle / "figure4_caption.tex").write_text(caption + "\n", encoding="utf-8")
    (bundle / "README.md").write_text(
        "# Figure 4: standardized PV at the reward port\n\n"
        "Complete Figure 4, 180 × 96 mm. Panels a–d/f retain the accepted numerical inputs and source drawing functions. "
        "Palette B (cividis) applies to all continuous maps/colorbars in a/c/e; orange data remain. The author-selected light gray (#C4C4C4) marks neuron-ID shuffles in b/e and matched spatial controls in c/d. Both the unfilled orange reward square and gray control squares in c have the same thin white outer edge for contrast. Panel f significance stars are black. "
        "Panel e uses the final v4 S analysis, not the earlier palette-preview analysis.\n\n"
        "## Rebuild from the frozen bundle\n\n"
        "`python source_code/build_figure4_standardized_pv.py --bundle . --output-dir rebuilt`\n\n"
        "Run from this folder, or pass absolute paths. Python requires NumPy, pandas, SciPy, Matplotlib, PyMuPDF, Pillow and statsmodels; "
        "the established exporter uses installed Chrome. No raw recordings, CIMaps, repository checkout or analysis rerun is needed. "
        "The frozen ca_analysis runtime is verified against its archive checksum before use.\n\n"
        "`plot_ready` contains heatmaps, row identities, pair counts, pair/animal PV profiles, plotted intervals and statistics. "
        "The maps are peak-normalized for display; the PV maps are standardized separately across spatial bins. "
        "`provenance` binds these compact inputs to the original maps and accepted sources. The full caption retains a–d/f text and replaces e. "
        "Input hashes are checked before every rebuild. `RENDER_VALIDATION.json` records artist parity, statistics and export checks. "
        "For the retained darker-gray alternative, add `--control-color '#454545' --output-dir darker_gray_option`. "
        "The default is the selected light gray; both versions have black panel f stars.\n",
        encoding="utf-8")
    inputs = [p for top in ["source_code", "reference_record", "plot_ready", "provenance"] for p in (bundle / top).rglob("*") if p.is_file()]
    dump(bundle / "INPUT_HASHES.json", {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(inputs)})


def route_panel(bundle, size):
    z = np.load(bundle / "plot_ready/panel_e_heatmaps.npz")
    curve = pd.read_csv(bundle / "plot_ready/panel_e_curve.csv")
    numbers = pd.read_csv(bundle / "plot_ready/panel_e_statistics.csv")
    fig = plt.figure(figsize=size)
    for key, group, title, x0, reverse in [
        ("h2r", "home_to_reward", "Home to reward", .075, False),
        ("r2h", "reward_to_home", "Reward to home", .555, True)]:
        axs = [fig.add_axes([x0+i*.130, .20, .110, .64]) for i in range(2)]
        for ax, suffix, label in zip(axs, ["day_i", "day_i90"], ["Day i", "Day i + 90"]):
            m = z[key + "_" + suffix]
            im = ax.imshow(m[:, ::-1] if reverse else m, aspect="auto", origin="lower", cmap="cividis",
                           vmin=0, vmax=.6, interpolation="antialiased", rasterized=True)
            ax.set_title(label, fontsize=5.8, pad=3)
            bins = [21, 15, 10, 5, 0] if reverse else [0, 5, 10, 15, 21]
            ax.set_xticks([21-b if reverse else b for b in bins], [str(b) for b in bins], fontsize=4.8)
            ax.set_xlabel(title, fontsize=5)
            ax.tick_params(labelsize=4.8, pad=1)
        axs[0].set_ylabel("Neurons", fontsize=5.2)
        axs[1].tick_params(labelleft=False)
        ax = fig.add_axes([x0+2*.130+.009, .20, .100, .64])
        g = curve[curve.group.eq(group)].sort_values("bin", ascending=False)
        yy = 21-g.bin.to_numpy()
        ax.fill_betweenx(yy, g.lo, g.hi, color=ORANGE, alpha=.22, lw=0)
        ax.plot(g["mean"], yy, color=ORANGE, lw=1.3, label="Data")
        ax.plot(g.shuffle, yy, color=GRAY, lw=.9, ls=":", label="Neuron ID shuffle")
        ax.set(xlim=(-.04, .36), ylim=(-.5, 21.5))
        ax.set_xticks([0, .1, .2, .3], ["0", "0.1", "0.2", "0.3"], fontsize=4.8)
        ax.set_yticks([21, 0], ["Home", "Reward"], fontsize=4.8)
        ax.yaxis.tick_right(); ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_xlabel("Standardized PV\ncorrelation", fontsize=5)
        ax.spines[["top", "left"]].set_visible(False)
        ax.legend(frameon=False, fontsize=4.6, loc="lower center", bbox_to_anchor=(.5, 1), handlelength=1.6)
        p = numbers[(numbers.group == group) & (numbers.item == "port minus elsewhere")].iloc[0].q_BH
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(float(g.loc[g.bin.eq(21), "mean"].iloc[0])+.03, 1.4, star, ha="left", va="center",
                fontsize=6.2, fontweight="bold", color="#202020")
    cax = fig.add_axes([.955, .34, .008, .4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Peak-normalized activity", fontsize=5, labelpad=2)
    cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
    z.close()
    return fig


def render(bundle, out):
    hashes = json.loads((bundle / "INPUT_HASHES.json").read_text(encoding="utf-8"))
    assert all(sha(bundle / p) == h for p, h in hashes.items()), "Frozen input hash mismatch"
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "individual_panels"; panels.mkdir(exist_ok=True)
    helper = load("figure4_local_rebuild", bundle / "source_code/rebuild_current_main_figures.py")
    helper.VOR_ROOT = bundle / "source_code"
    helper.REPO_ROOT = bundle
    record = bundle / "reference_record"
    pub = helper._load_publication_renderer(record, "figure4_local_publication")
    mod = helper._load_module("figure4_local_panels", record / "source_code/render_figure4_day_source_comparison.py")
    pf = helper._load_module("figure4_local_f", record / "source_code/render_main_figure4_panel_f_ed10_k166.py")
    mod.configure_style()
    plt.rcParams.update({"svg.hashsalt": "figure4-standardized-pv-v4", "savefig.bbox": None})
    ready = record / "plot_ready"
    maps = pd.read_csv(ready / "panel_c_example_map_rows.csv")
    d = pd.read_csv(ready / "panel_d_native_reference_rows.csv", dtype={"animal": str})
    matched = pd.read_csv(ready / "panel_f_matched_pair_means.csv", dtype={"animal_id": str})
    numbers = pd.read_csv(ready / "figure4_day_source_statistics.csv")
    shuffle = pd.read_csv(record / "source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_pair_summary.csv", dtype={"animal_id": str})
    shuffle_stats = json.loads((record / "source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_statistics.json").read_text())
    style = pub.PublicationStyle(dpi=600)
    pub.FIGURE4_SCHEMATIC_DATA_ROOT = record / "illustration_data"
    figures = {}
    with plt.rc_context(style.rcparams()):
        figures["a"], schematic = pub._render_figure4_method_schematic(mod.SLOTS["a"][2:], style)
    native_schematic_bins(figures["a"])
    mod.configure_style()
    figures["b"] = mod.paired_panel(shuffle.rename(columns={"neuron_id_shuffle_error": "exact_label_permutation_error"}),
        pd.Series({"cluster_p": shuffle_stats["cluster_regression"]["cluster_p"]}), variant="d1", panel="b", size=mod.SLOTS["b"][2:])
    figures["c"] = mod.map_panel(maps.loc[maps.variant.eq("d1")], variant="d1", size=mod.SLOTS["c"][2:], source_csv=out / "panel_c_source_rows.csv")
    figures["d"] = mod.paired_panel(d.loc[d.variant.eq("d1v1") & d.included_reward_location_control.astype(bool)],
        numbers.loc[numbers.analysis.eq("d1_native_reward_control_advantage")].iloc[0], variant="d1", panel="d", size=mod.SLOTS["d"][2:])
    figures["e"] = route_panel(bundle, mod.SLOTS["e"][2:])
    mod.configure_style()
    figures["f"] = pf.render_panel(matched, size=mod.SLOTS["f"][2:])
    for f in figures.values(): f.canvas.draw()
    before = {k: signature(f) for k, f in figures.items()}
    f_stars = [t for ax in figures["f"].axes for t in ax.texts if t.get_text() in ["*", "**", "***"]]
    assert len(f_stars) == 1
    f_stars[0].set_color("#202020")
    # Recolor already drawn accepted artists so array/geometry parity is checked
    # against their original source output, not against a modified helper.
    from matplotlib.colors import to_rgba
    reference_rgb = np.asarray(to_rgba("#7f7f7f"))[:3]
    recolored = {}
    for k in "bd":
        n = 0
        for ax in figures[k].axes:
            for collection in ax.collections:
                faces = collection.get_facecolors()
                if len(faces) and np.allclose(faces[:, :3], reference_rgb):
                    faces[:, :3] = to_rgba(GRAY)[:3]
                    collection.set_facecolors(faces); n += 1
            for patch in ax.patches:
                face = patch.get_facecolor()
                if np.allclose(face[:3], reference_rgb):
                    patch.set_facecolor((*to_rgba(GRAY)[:3], face[3])); n += 1
        assert n == 2, "One reference scatter and one mean bar are expected"
        recolored[k] = n
    control_outlines = 0
    reward_outlines = 0
    import matplotlib.patheffects as path_effects
    for ax in figures["c"].axes:
        for patch in ax.patches:
            if np.allclose(patch.get_edgecolor(), to_rgba("#7f7f7f")):
                patch.set_edgecolor(GRAY)
                patch.set_path_effects([path_effects.Stroke(linewidth=patch.get_linewidth()+.8, foreground="white"), path_effects.Normal()])
                control_outlines += 1
            elif np.allclose(patch.get_edgecolor(), to_rgba(ORANGE)):
                patch.set_path_effects([path_effects.Stroke(linewidth=patch.get_linewidth()+.8, foreground="white"), path_effects.Normal()])
                reward_outlines += 1
    assert control_outlines == 3, "The three matched-control outlines must be recolored"
    assert reward_outlines == 1, "The orange reward square must receive the same white under-outline"
    fixed_svg = {k: svg(figures[k]) for k in "bdf"}
    counts = {}
    for k in "ace":
        images = figures[k].findobj(match=AxesImage)
        counts[k] = len(images)
        for im in images:
            cmap = plt.get_cmap("cividis").copy(); cmap.set_bad(im.get_cmap().get_bad()); im.set_cmap(cmap)
    after = {k: signature(f) for k, f in figures.items()}
    assert before == after, "Palette change altered scientific artists"
    # Raster density must be set at SVG creation: a later 600-dpi page export
    # cannot recover neuron rows already collapsed into a 100-dpi SVG image.
    payloads = {**fixed_svg, **{k: svg(figures[k], dpi=600 if k == "e" else None) for k in "ace"}}
    for k, f in figures.items():
        (panels / f"figure4_panel_{k}.svg").write_bytes(payloads[k])
        f.savefig(panels / f"figure4_panel_{k}.pdf", bbox_inches=None, dpi=600 if k == "e" else None,
                  metadata={"CreationDate": None, "ModDate": None})
        for dpi in [300, 600]:
            f.savefig(panels / f"figure4_panel_{k}_{dpi}dpi.png", dpi=dpi, bbox_inches=None)
    stem = out / "figure4_standardized_pv"
    helper._compose_main4(pub, mod, payloads, stem.with_suffix(".svg"))
    outputs = helper._export_svg_bundle(stem.with_suffix(".svg"), stem)
    with fitz.open(stem.with_suffix(".pdf")) as doc:
        assert len(doc) == 1
        page = doc[0]; dims = [page.rect.width*25.4/72, page.rect.height*25.4/72]
        assert np.allclose(dims, [180, 96], atol=.35)
        text = page.get_text()
        assert text.count("Standardized PV") == 2
        assert "Population vector" not in text and "Density" not in text
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out / "figure4_standardized_pv_300dpi.png", dpi=(300, 300))
    dump(out / "SCIENTIFIC_ARTISTS.json", after)
    dump(out / "SCHEMATIC_SOURCE.json", schematic)
    dump(out / "RENDER_VALIDATION.json", dict(status="PASS", rule_version=RULE, palette="B / cividis",
        input_hashes_verified=len(hashes), figure_mm=dims, continuous_image_counts=counts,
        unchanged_science="a–d/f: exact accepted drawing functions and frozen data; all array/mask/line/collection/label/geometry signatures unchanged by palette",
        approved_control_color=GRAY, recolored_reference_artists=recolored, matched_control_outlines=control_outlines,
        white_under_outline={"controls": control_outlines, "reward": reward_outlines, "extra_width_pt": .8, "centers_remain_unfilled": True},
        approved_panel_f_star_color="#202020",
        panel_a_template_export="Five native 15x15 images, no interpolation; exact original bin values and affine geometry",
        panel_e_raster_dpi=600,
        colormap_only_svg_parity="b/d/f SVGs unchanged by the cividis step; the separately approved gray control recolor changes b/d before this check",
        heatmap_order="Day-i peak sort shared across dates; peak at port bottom in both directions",
        curve_orientation="Port at bottom, home at top; heatmaps read left-to-right in direction of travel",
        panel_e_statistics=pd.read_csv(bundle / "plot_ready/panel_e_statistics.csv").fillna("").to_dict("records"),
        outputs={p.relative_to(out).as_posix(): sha(p) for p in outputs}))
    assert all(sha(bundle / p) == h for p, h in hashes.items())
    plt.close("all")
    print(json.dumps({"status": "PASS", "output": str(out), "figure_mm": dims}), flush=True)


def main():
    global GRAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--control-color", default="#C4C4C4", choices=["#454545", "#C4C4C4"])
    args = parser.parse_args()
    GRAY = args.control_color
    bundle = (args.bundle or args.repo_root / REL).resolve()
    if args.freeze:
        freeze(bundle, args.repo_root.resolve())
    render(bundle, (args.output_dir or bundle).resolve())


if __name__ == "__main__":
    main()
