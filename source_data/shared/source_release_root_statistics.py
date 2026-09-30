"""Source-data statistical replay for ED6, ED8, ED10 and supplementary tables."""
from __future__ import annotations
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cr1(y, groups, x=None):
    y = np.asarray(y, float)
    X = np.ones((len(y), 1)) if x is None else sm.add_constant(np.asarray(x, float), has_constant='add')
    fit = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': np.asarray(groups).astype(str), 'use_correction': True})
    estimate, se = float(fit.params[-1]), float(fit.bse[-1])
    df = len(np.unique(groups)) - 1
    crit = stats.t.ppf(.975, df)
    return dict(estimate=estimate, cluster_se=se, t_value=estimate/se, df=df,
                p_two_sided=float(2*stats.t.sf(abs(estimate/se), df)), ci_low=estimate-crit*se, ci_high=estimate+crit*se)


def compare_table(actual, reference, keys, label, out):
    actual.to_csv(out / (label + '.csv'), index=False)
    left, right = actual.set_index(keys).sort_index(), reference.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise AssertionError(label + ' row keys differ')
    checked, errors = [], []
    for col in reference.columns:
        if col in keys or col not in actual or not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        a, b = pd.to_numeric(left[col]).to_numpy(float), pd.to_numeric(right[col]).to_numpy(float)
        atol = 1e-15 if 'p_' in col or col.endswith('_p') or 'q_' in col or col.startswith('p') else 2e-11
        if not np.allclose(a, b, rtol=1e-8, atol=atol, equal_nan=True):
            errors.append({'column': col, 'max_abs_error': float(np.nanmax(np.abs(a-b)))})
        checked.append(col)
    if errors:
        raise AssertionError(label + ': ' + json.dumps(errors))
    return dict(analysis=label, status='PASS', n_tests=len(actual), compared_columns=checked,
                statistical_inputs='unit-level derived rows; saved statistics used only for comparison')


def ed6(release, out):
    source = release / 'extended_data_figure_6/source'
    inputs, plot = source / 'statistical_inputs', source / 'plot_ready'
    module = load(inputs / 'fixed_stats_helpers.py', 'release_ed6_stats')
    pair = pd.read_csv(inputs / 'ed07_current59_pair_metric_inclusion_audit.csv', dtype={'animal': str})
    for col in ['included_in_slope_test', 'included_in_experience_test']:
        pair[col] = pair[col].astype(str).str.lower().eq('true')
    results = [compare_table(module._build_stats(pair), pd.read_csv(plot / 'ed07_current59_clustered_stats.csv'),
                            ['metric', 'window_type', 'window_size', 'estimand'], 'ED6_fixed_reference_CR1_BH18', out)]
    pair = pd.read_csv(inputs / 'ed07_pivot_maze_time_pair_early_slope_inclusion_current59.csv', dtype={'animal': str})
    pair['pair_included'] = pair.pair_included.astype(str).str.lower().eq('true')
    rows = []
    keys = ['metric', 'metric_key', 'metric_label', 'support_policy', 'window_size_min']
    for key, all_rows in pair.groupby(keys, sort=False):
        selected = all_rows[all_rows.pair_included]
        for day in ['day1', 'day2']:
            rows.append(dict(zip(keys,key), day=day, n_eligible_pair_rows=len(all_rows), n_pair_rows=len(selected),
                             n_animal_clusters=selected.animal.nunique(), **cr1(selected[day+'_slope'], selected.animal)))
    actual = pd.DataFrame(rows)
    actual['q_bh_two_sided'] = actual.groupby('day').p_two_sided.transform(lambda s: multipletests(s, method='fdr_bh')[1])
    results.append(compare_table(actual, pd.read_csv(plot / 'ed07_pivot_day_specific_early_slope_stats_current59.csv'),
                                 ['metric', 'window_size_min', 'day'], 'ED6_pivot_CR1_BH9', out))
    return results


def exact_flip(values):
    values=np.asarray(values,float)
    signs=np.asarray(list(itertools.product([-1.,1.], repeat=len(values))))
    return float(np.mean(np.abs(signs@values) >= abs(values.sum()) - 1e-12))


def ed8(release, out):
    source = release / 'extended_data_figure_8/source'
    sd = source / 'source_data'
    reference = pd.read_csv(sd / 'accepted_stats.csv')
    b = pd.read_csv(sd / 'accepted_b.csv', dtype={'animal': str})
    rows=[]
    for (variant, method), pair in b.groupby(['variant','resample_method']):
        delta=(pair.matched_control_error-pair.reward_proxy_error).to_numpy(float)
        mean=delta.mean(); half=stats.sem(delta)*stats.t.ppf(.975,len(delta)-1)
        # Accepted decoding family is one-sided (control error > reward error).
        rows.append(dict(panel_role='B',subestimand=variant+':'+method,estimate=mean,ci95_low=mean-half,ci95_high=mean+half,p_value=exact_flip(delta)/2))
    actual=pd.DataFrame(rows); actual.p_value=multipletests(actual.p_value,method='fdr_bh')[1]
    results=[compare_table(actual,reference[reference.panel_role.eq('B')],['subestimand'],'ED8b_equal_mouse_exact_BH4',out)]
    c=pd.read_csv(sd/'accepted_c.csv',dtype={'animal':str})
    rows=[]
    for variant,pair in c.groupby('pair_variant'):
        r=cr1(pair.matched_contrast,pair.animal)
        rows.append(dict(panel_role='C',subestimand=variant,estimate=r['estimate'],ci95_low=r['ci_low'],ci95_high=r['ci_high'],p_value=r['p_two_sided']))
    actual=pd.DataFrame(rows); actual.p_value=multipletests(actual.p_value,method='fdr_bh')[1]
    results.append(compare_table(actual,reference[reference.panel_role.eq('C')],['subestimand'],'ED8c_pair_CR1_BH2',out))
    # Reuse the source-frozen route producer's complete animal aggregation/test
    # gate without rendering. It verifies raw exact p-values and every BH family.
    module=load(source/'source_code/build_ed8_standardized_pv.py','release_ed8_stats')
    (out/'source_data').mkdir(exist_ok=True)
    shutil.copy2(sd/'candidate_paired_display_summary.csv',out/'source_data/candidate_paired_display_summary.csv')
    data,pair,num,animal,rep,rn,xp,xn=module.read(source)
    _,records=module.verify(out,pair,num,animal,rep,rn,xp,xn)
    results.append(dict(analysis='ED8_standardized_route_exact_signflip_and_BH',status='PASS',n_tests=len(records),
                        compared_columns=['mean','lo','hi','p_exact','adjusted_p'],statistical_unit='mouse after averaging pairs within mouse'))
    return results


def ed10(release,out):
    source=release/'extended_data_figure_10/source'
    record=source/'frozen_source/extended_data_fig10_round18_inputs'
    plot=record/'plot_ready'
    frame=pd.read_csv(plot/'CURRENT_FIXED_NEURON_PAIR_ROWS.csv',dtype={'animal_id':str},float_precision='round_trip')
    # Matrix display Pearson r and raw nominal p are reconstructed, not imported.
    table=pd.read_csv(plot/'POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv')
    cols=dict(reward_location='reward_error_fixed2000',whole_decoder='whole_error_fixed2000',place_map='place_map',structure='structure',pv_map='pv_map',rate='rate')
    outcomes=dict(whole_session_reward_approach_speed_advantage='log_reward_over_pooled',first_long_bout_time='first_long_bout_time_s',exploration_efficiency='exploration_efficiency_raw')
    rows=[]
    for ref in table.itertuples():
        x,y=frame[cols[ref.block]],frame[outcomes[ref.outcome]]
        r,p=stats.pearsonr(frame[ref.block],y)
        fit=cr1(y,frame.animal_id,x)
        rows.append(dict(block=ref.block,outcome=ref.outcome,pooled_pearson_r=r,pairwise_nominal_p_descriptive=p,cr1_p_two_sided=fit['p_two_sided']))
    actual=pd.DataFrame(rows)
    actual['cr1_bh3_adjusted_p']=actual.groupby('block').cr1_p_two_sided.transform(lambda s:multipletests(s,method='fdr_bh')[1])
    actual['cr1_bh6_adjusted_p']=actual.groupby('outcome').cr1_p_two_sided.transform(lambda s:multipletests(s,method='fdr_bh')[1])
    results=[compare_table(actual,table,['block','outcome'],'ED10a_Pearson_CR1_BH3_and_BH6',out)]
    expected=pd.read_csv(source/'statistical_inputs/panel_a_expected_multiplicity.csv')
    results.append(compare_table(actual,expected[['block','outcome','cr1_p_two_sided','cr1_bh3_adjusted_p','cr1_bh6_adjusted_p']],
                                 ['block','outcome'],'ED10a_CR1_nominal_and_adjusted_p_parity',out))
    actual.to_csv(out/'ED10a_Pearson_CR1_BH3_and_BH6.csv',index=False)
    helpers=load(source/'statistical_inputs/profile_helpers.py','release_ed10_profile_stats')
    commonality,profile,lomo=helpers.reward_level_profile_analysis(frame)
    results.append(compare_table(commonality,pd.read_csv(plot/'REWARD_LEVEL_COMMONALITY.csv'),['analysis'],'ED10b_commonality',out))
    results.append(compare_table(profile,pd.read_csv(plot/'REWARD_LEVEL_PROFILE_TESTS.csv'),['claim','specification','weighting'],'ED10b_CR1_exact_cluster_score_maxT3',out))
    lomo.to_csv(out/'ED10b_leave_one_mouse_out.csv',index=False)
    bridge=pd.read_csv(plot/'level_policy_bridge_join_corrected36.csv',dtype={'animal_id':str})
    reference=pd.read_csv(plot/'level_policy_bridge_stats_corrected36.csv')
    rows=[]
    for ref in reference.itertuples():
        x,y=bridge[ref.x],bridge[ref.y]
        r,p=stats.pearsonr(x,y);rho,sp=stats.spearmanr(x,y)
        fit=cr1((y-y.mean())/y.std(ddof=0),bridge.animal_id,(x-x.mean())/x.std(ddof=0))
        rows.append(dict(x=ref.x,y=ref.y,n_pairs=len(bridge),n_animals=bridge.animal_id.nunique(),pearson_r=r,pearson_p=p,
                         spearman_rho=rho,spearman_p=sp,cluster_beta_standardized=fit['estimate'],cluster_se=fit['cluster_se'],
                         cluster_t=fit['t_value'],cluster_df=fit['df'],cluster_p=fit['p_two_sided'],cluster_p_t_clusters_minus_1=fit['p_two_sided'],
                         cluster_p_normal=2*stats.norm.sf(abs(fit['t_value']))))
    actual=pd.DataFrame(rows)
    family=pd.read_csv(source/'statistical_inputs/policy_window_pair_rows.csv',dtype={'animal_id':str})
    family_rows=[]
    for window,pair in family.groupby('window_id'):
        for ycol in ['Lf','Lo_depth_weighted']:
            selected=pair[['level_mean_all',ycol,'animal_id']].dropna()
            fit=cr1(selected[ycol],selected.animal_id,selected.level_mean_all)
            family_rows.append(dict(window_id=window,policy=ycol,p_raw=fit['p_two_sided']))
    family_stats=pd.DataFrame(family_rows)
    family_stats['p_BH18']=multipletests(family_stats.p_raw,method='fdr_bh')[1]
    family_stats.to_csv(out/'ED10c_policy_window_family_BH18.csv',index=False)
    actual['cluster_q_bh_window_family']=np.nan
    for policy in ['Lf','Lo_depth_weighted']:
        actual.loc[actual.y.eq(policy),'cluster_q_bh_window_family']=family_stats.query('window_id == "fixed_30min" and policy == @policy').p_BH18.iloc[0]
    results.append(compare_table(actual,reference,['x','y'],'ED10c_policy_CR1_and_window_BH18',out))
    return results


def run(release_root,output_dir):
    release_root,output_dir=Path(release_root),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    analyses=[];gaps=[]
    for name,fn in [('ED6',ed6),('ED8',ed8),('ED10',ed10)]:
        out=output_dir/name;out.mkdir(exist_ok=True)
        try:
            analyses.extend(fn(release_root,out))
        except Exception as error:
            import traceback
            gaps.append(dict(artifact=name,error=str(error),traceback=traceback.format_exc()))
    result=dict(status='PASS' if not gaps else 'FAIL',analyses=analyses,gaps=gaps)
    (output_dir/'STATISTICS_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
