"""Standalone level counterpart of current Figure2d; no manuscript figure writes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'output/figures/level_population_vector_distribution_preview_2026-09-26'
LEVEL_SOURCE=ROOT/'output/figures/ed9_three_row_level_pv_2026-09-25/source_data/population_vector_shuffle'
FIG2=ROOT/'docs/figure_version_records/main_figure2_2026-09-25_round13_population_vector_first'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def freeze(output):
    frozen=output/'frozen'
    frozen.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((LEVEL_SOURCE/'manifest.json').read_text(encoding='utf-8'))
    names=['pair_summary.csv','pair_by_depth.csv']
    names += [str(p.relative_to(LEVEL_SOURCE/'frozen')).replace('\\','/') for p in sorted((LEVEL_SOURCE/'frozen/intermediates').glob('*.npz'))]
    rows=[]
    for name in names:
        src=LEVEL_SOURCE/'frozen'/name
        assert sha(src)==manifest['outputs']['frozen/'+name]['sha256']
        dst=frozen/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        rows.append({'path':name,'source':str(src),'sha256':sha(dst)})
    for src,name in [(FIG2/'source_data/distribution/figure2_panel_e_pv_distribution_corrected_842_no_presort_exclusions_36_no_1616_short.npz','figure2d_reference.npz'),
                     (FIG2/'source_code/render_main_figs_v3_publication.py','figure2_renderer_reference.py')]:
        shutil.copy2(src,frozen/name)
        rows.append({'path':name,'source':str(src),'sha256':sha(frozen/name)})
    write(frozen/'INPUTS.json',rows)
    return frozen

def build(inputs,output):
    output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((inputs/'INPUTS.json').read_text(encoding='utf-8'))
    assert all(sha(inputs/r['path'])==r['sha256'] for r in audit)
    pairs=pd.read_csv(inputs/'pair_summary.csv')
    original=pd.read_csv(inputs/'pair_by_depth.csv').set_index(['pair_key','depth'])
    with np.load(inputs/'figure2d_reference.npz') as ref:
        edges=np.asarray(ref['hist_edges'],dtype=float)
    assert len(edges)==101 and edges[0]==-1 and edges[-1]==1
    counts=np.zeros(100,dtype=np.int64)
    rows=[];by_level=[];supports=[];pooled_null=[];max_error=0
    for index,pair in enumerate(pairs.itertuples(index=False),1):
        path=inputs/'intermediates'/f'{index:02d}_{pair.src_session}_to_{pair.dst_session}.npz'
        with np.load(path) as z:
            mask=z['common_mask'].astype(bool)
            assert np.array_equal(mask,z['train_mask'] & z['test_mask'])
            raw=np.asarray(z['neuron_id_shuffle_by_depth'],dtype=float)
            assert raw.shape==(10000,7)
            assert np.isfinite(raw[:,mask]).all() and np.isnan(raw[:,~mask]).all()
            obs=np.diag(z['matrix'])
            for d in range(7):
                old=original.loc[(pair.pair_key,d)]
                assert bool(old.supported)==bool(mask[d])
                supports.append({'pair_key':pair.pair_key,'animal_id':pair.animal_id,'level':d,
                                 'train_supported':bool(z['train_mask'][d]),'test_supported':bool(z['test_mask'][d]),
                                 'included':bool(mask[d]),'n_registered_neurons':pair.n_neurons})
                if not mask[d]: continue
                # Independent Pearson calculation across neurons for this level.
                r=float(np.corrcoef(z['train_map'][d],z['test_map'][d])[0,1])
                max_error=max(max_error,abs(r-obs[d]),abs(obs[d]-float(old.same_depth)))
                assert max_error<1e-12
                values=raw[:,d]
                hist=np.histogram(values,bins=edges)[0]
                assert hist.sum()==10000
                counts+=hist
                rows.append({'pair_key':pair.pair_key,'animal_id':pair.animal_id,'level':d,
                             'population_vector_correlation':float(obs[d]),'n_registered_neurons':pair.n_neurons,
                             'source_reward_condition':pair.source_reward_condition,
                             'target_reward_condition':pair.target_reward_condition,
                             'shuffle_count':len(values),'shuffle_mean':float(values.mean())})
                for k,n in enumerate(hist):
                    by_level.append({'pair_key':pair.pair_key,'level':d,'bin_index':k,'shuffle_count':int(n)})
            pooled_null.append(raw[:,mask].ravel())
    observations=pd.DataFrame(rows)
    obs=observations.population_vector_correlation.to_numpy()
    assert len(pairs)==36 and pairs.animal_id.nunique()==14 and len(obs)==248
    assert counts.sum()==2480000 and len(supports)==252
    null=np.concatenate(pooled_null)
    assert np.array_equal(counts,np.histogram(null,bins=edges)[0])
    observed_counts=np.histogram(obs,bins=edges)[0]
    widths=np.diff(edges); centers=(edges[:-1]+edges[1:])/2
    obs_density=observed_counts/(len(obs)*widths)
    null_density=counts/(counts.sum()*widths)
    assert np.isclose(sum(obs_density*widths),1) and np.isclose(sum(null_density*widths),1)
    observations['histogram_weight']=1/len(obs)
    observations.to_csv(output/'observed_pair_level_values.csv',index=False)
    pd.DataFrame(supports).to_csv(output/'support_audit.csv',index=False)
    pd.DataFrame(by_level).to_csv(output/'shuffle_histogram_by_pair_level.csv',index=False)
    pd.DataFrame({'left':edges[:-1],'right':edges[1:],'data_count':observed_counts,'shuffle_count':counts,
                  'data_density':obs_density,'shuffle_density':null_density}).to_csv(output/'histogram.csv',index=False)
    np.savez_compressed(output/'plot_data.npz',obs=obs,shuffle_hist=counts,hist_edges=edges)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.3,
                        'ytick.labelsize':6.3,'axes.linewidth':.65,'axes.spines.top':False,
                        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
                        'svg.hashsalt':'level_population_vector_distribution_preview'})
    fig=plt.figure(figsize=(50/25.4,43/25.4),dpi=100)
    ax=fig.add_axes([9/50,10/43,38/50,30/43])
    shuffle_bars=ax.bar(centers,null_density,width=widths,color='#8a8a8a',alpha=.62,edgecolor='none',zorder=1)
    data_bars=ax.bar(centers,obs_density,width=widths,color='#7976A2',alpha=.72,edgecolor='none',zorder=2)
    ax.set(xlim=(-.45,1),ylim=(0,max(null_density.max(),obs_density.max())*1.10),
           xlabel='Population vector correlation\n(level)',ylabel='Density',xticks=[0,.5,1])
    ax.tick_params(width=.65,length=2,pad=1.5)
    ax.xaxis.labelpad=2;ax.yaxis.labelpad=2
    labels=[ax.text(.05,.88,'Shuffle',transform=ax.transAxes,ha='left',va='top',fontsize=7,
                    color='#6f6f6f',fontweight='bold'),
            ax.text(.64,.40,'Data',transform=ax.transAxes,ha='left',va='center',fontsize=7,
                    color='#7976A2',fontweight='bold')]
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    overlaps=[]
    for t in labels:
        box=t.get_window_extent(renderer).padded(1*fig.dpi/72)
        for bars in (shuffle_bars,data_bars):
            for bar in bars:
                if bar.get_height()>0 and box.overlaps(bar.get_window_extent(renderer)):
                    overlaps.append(t.get_text())
    assert not overlaps,overlaps
    text_in_bounds=all(fig.bbox.contains(*t.get_window_extent(renderer).get_points()[i])
                       for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()
                       for i in (0,1))
    assert text_in_bounds
    stem=output/'level_population_vector_distribution'
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(stem.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    for dpi in (300,600):
        fig.savefig(output/f'{stem.name}_{dpi}dpi.png',dpi=dpi,facecolor='white')
    with Image.open(output/f'{stem.name}_600dpi.png') as im:
        im.save(stem.with_suffix('.tiff'),compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)
    summary={'status':'PASS','counterpart':'Current Figure2 Round13 panel d population-vector distribution',
             'observations':248,'pairs':36,'animals':14,'null_draws_per_pair_level':10000,
             'null_values':int(counts.sum()),'observed_pooled_mean':float(obs.mean()),
             'null_pooled_mean':float(null.mean()),'equal_pair_mean_for_comparison':float(pairs.same_depth.mean()),
             'maximum_independent_pearson_error':max_error,'unsupported_pair_levels':4,
             'level_support_counts':observations.groupby('level').size().to_dict(),
             'histogram_bin_width':.02,'visible_xlim':[-.45,1],
             'observations_outside_visible_xlim':int(np.sum((obs<-.45)|(obs>1))),
             'shuffle_values_outside_visible_xlim':int(np.sum((null<-.45)|(null>1))),
             'density_integrals':[float(sum(obs_density*widths)),float(sum(null_density*widths))],
             'rendered_label_mark_overlaps':overlaps,'all_text_inside_canvas':text_in_bounds,
             'new_permutations':0,'p_value_added':False,'current_figures_modified':False}
    write(output/'VALIDATION.json',summary)
    (output/'README.md').write_text('''# Level population-vector distribution preview

Standalone counterpart of current Figure 2d (Round13, population-vector row first).
Purple: 248 supported same-level Pearson correlations across registered neurons,
from 36 approximately 3-month session pairs in 14 mice. Gray: 2,480,000 saved
neuron-ID-shuffled correlations (10,000 per supported pair-level observation).
Seven ordinary maze levels 0-6; four pairs lack level 6, which is omitted rather
than zero-filled. Same S-rate-qualified registered population and raw C/exposure
level maps as the current level population-vector heatmap and ED9a.

This pools supported pair-level observations equally, analogous to Figure2d's
pooled supported pair-spatial-bin distribution. Pairs supporting seven levels
therefore contribute seven values and the four other pairs contribute six.
This is distinct from ED9a, which averages levels within each pair and pairs equally
before displaying the permutation distribution of that mean. No neuron-wise level
normalization, new permutations, smoothing, significance test or new inference was
introduced. Levels and shuffle draws are not independent biological replicates.
Five initially unrewarded source sessions remain included; other sources and all
targets are rewarded at node 94. Reward conditions are in the observation table.

The same 100 bin edges [-1,1], visible range [-0.45,1], gray shuffle and overlapping
bars/alpha as Figure2d are used; level purple follows the revised level figures.
Each density integrates to one. Source values, all 252 support decisions and
per-pair-level shuffle histograms are exported. Frozen inputs include all cached
permutations, raw maps, IDs, masks and numerical references with input hashes.

Rebuild: `python rebuild_preview.py --inputs frozen --output rebuilt`

This is a preview only. Current assembled figures, version records and ltm-figs
PNGs were not modified.
''',encoding='utf-8')
    shutil.copy2(Path(__file__),output/'rebuild_preview.py') if Path(__file__).resolve()!=(output/'rebuild_preview.py').resolve() else None
    write(output/'OUTPUT_HASHES.json',{str(p.relative_to(output)).replace('\\','/'):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='OUTPUT_HASHES.json'})
    print(json.dumps(summary))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    inputs=args.inputs.resolve() if args.inputs else (output/'frozen' if (output/'frozen/INPUTS.json').exists() else freeze(output))
    build(inputs,output)
