"""Source-data statistical replay for ED6, ED8 and ED10."""
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


# Each contract lists the quantities this replay actually reconstructs. Reference
# tables can also contain labels or historical summaries outside that contract.
COMPARISON_COLUMNS = {
    'ED6_fixed_reference_CR1_BH12': (
        'n_pair_rows', 'n_animal_clusters', 'estimate', 'cluster_se', 't_value',
        'df', 'p_two_sided', 'ci_low', 'ci_high', 'q_bh12_two_sided'),
    'ED6_available_curve_CR1_BH3': (
        'n_recording_cycles', 'n_animals', 'estimate', 'cluster_se', 't_value',
        'df', 'p_two_sided', 'ci_low', 'ci_high', 'mean_day_i_within_curve_sd',
        'mean_day_i1_within_curve_sd', 'p_bh3_two_sided'),
    'ED8b_equal_mouse_exact_BH4': ('estimate', 'ci95_low', 'ci95_high', 'p_value'),
    'ED8c_pair_CR1_BH2': ('estimate', 'ci95_low', 'ci95_high', 'p_value'),
    'ED10a_Pearson_animal_clustered_BH3': ('pooled_pearson_r', 'pairwise_nominal_p_descriptive'),
    'ED10a_CR1_nominal_and_adjusted_p_parity': ('cr1_p_two_sided', 'cr1_bh3_adjusted_p'),
    'ED10b_commonality': (
        'r2_baseline', 'r2_a', 'r2_b', 'r2_full', 'full_increment',
        'unique_a', 'shared', 'unique_b'),
    'ED10b_exact_animal_score_signflip_BH3': (
        'estimate', 'cr1_se', 'cr1_t', 'cr1_df', 'cr1_p', 'ci95_low', 'ci95_high',
        'n_pairs', 'n_animals', 'score_t', 'exact_p_two_sided', 'n_patterns', 'bh3_adjusted_p'),
    'ED10c_policy_animal_clustered_BH4': (
        'n_pairs', 'n_animals', 'pearson_r', 'cluster_beta_standardized',
        'cluster_se', 'cluster_t', 'cluster_df', 'cluster_p', 'cluster_bh4_adjusted_p'),
}


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
    columns = COMPARISON_COLUMNS.get(label)
    if not columns:
        raise AssertionError(label + ' has no numeric comparison contract')
    for side, table in [('regenerated', actual), ('reference', reference)]:
        missing = set(keys).union(columns).difference(table.columns)
        if missing or table.empty:
            raise AssertionError(f'{label}: {side} is empty or missing required columns: {sorted(missing)}')
    actual.to_csv(out / (label + '.csv'), index=False)
    left, right = actual.set_index(keys).sort_index(), reference.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise AssertionError(label + ' row keys differ')
    checked, errors = [], []
    for col in columns:
        a, b = pd.to_numeric(left[col]).to_numpy(float), pd.to_numeric(right[col]).to_numpy(float)
        probability = bool({'p', 'q'}.intersection(col.lower().split('_')))
        # Tiny probabilities require relative agreement; an ordinary absolute
        # tolerance could conceal several orders of magnitude of drift.
        atol = 1e-300 if probability else 2e-11
        same_nan = np.isnan(a) & np.isnan(b)
        close = same_nan | (np.isfinite(a) & np.isfinite(b) & (np.abs(a-b) <= atol + 1e-8*np.abs(b)))
        if not close.all():
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
    for col in ['included_in_slope_test']:
        pair[col] = pair[col].astype(str).str.lower().eq('true')
    results = [compare_table(module._build_stats(pair), pd.read_csv(plot / 'ed07_current59_clustered_stats.csv'),
                            ['metric', 'window_type', 'window_size', 'estimand'], 'ED6_fixed_reference_CR1_BH12', out)]
    pair = pd.read_csv(inputs / 'ed06_pair_metric_available_curve_summaries.csv', dtype={'animal': str})
    rows = []
    estimands = {'day_i1_minus_day_i_within_curve_sd': 'day_i1_minus_day_i_sd',
                 'day_i_linear_slope': 'day_i_slope_r_per_hour',
                 'day_i1_linear_slope': 'day_i1_slope_r_per_hour'}
    for estimand, column in estimands.items():
        for (metric, label), selected in pair.groupby(['metric', 'metric_label'], sort=False):
            row = dict(estimand=estimand, metric=metric, metric_label=label, value_column=column,
                       n_recording_cycles=len(selected), n_animals=selected.animal.nunique(),
                       **cr1(selected[column], selected.animal))
            if estimand == 'day_i1_minus_day_i_within_curve_sd':
                row.update(mean_day_i_within_curve_sd=selected.day_i_within_curve_sd.mean(),
                           mean_day_i1_within_curve_sd=selected.day_i1_within_curve_sd.mean())
            rows.append(row)
    actual = pd.DataFrame(rows)
    if len(actual) != 9 or not actual.groupby('estimand').size().eq(3).all():
        raise AssertionError('ED6 pivot requires three displayed metrics in each of three estimand families')
    actual['p_bh3_two_sided'] = actual.groupby('estimand').p_two_sided.transform(lambda s: multipletests(s, method='fdr_bh')[1])
    results.append(compare_table(actual, pd.read_csv(plot / 'ed06_available_curve_clustered_stats.csv'),
                                 ['estimand', 'metric'], 'ED6_available_curve_CR1_BH3', out))
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
        if not mean > 0:
            raise AssertionError("ED8 decoding one-sided tail requires a positive control-minus-reward effect")
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
    # Pearson r describes the displayed association. Inference uses an OLS slope
    # with animal-clustered standard errors, followed by BH3 within each row.
    # The ordinary Pearson p value is retained only as a descriptive comparison.
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
    results=[compare_table(actual,table,['block','outcome'],'ED10a_Pearson_animal_clustered_BH3',out)]
    expected=pd.read_csv(source/'statistical_inputs/panel_a_expected_multiplicity.csv')
    results.append(compare_table(actual,expected[['block','outcome','cr1_p_two_sided','cr1_bh3_adjusted_p']],
                                 ['block','outcome'],'ED10a_CR1_nominal_and_adjusted_p_parity',out))
    actual.to_csv(out/'ED10a_Pearson_animal_clustered_BH3.csv',index=False)
    helpers=load(source/'statistical_inputs/profile_helpers.py','release_ed10_profile_stats')
    commonality,profile=helpers.reward_level_profile_analysis(frame)
    results.append(compare_table(commonality,pd.read_csv(plot/'REWARD_LEVEL_COMMONALITY.csv'),['analysis'],'ED10b_commonality',out))
    results.append(compare_table(profile,pd.read_csv(plot/'REWARD_LEVEL_PROFILE_TESTS.csv'),['claim','specification','weighting'],'ED10b_exact_animal_score_signflip_BH3',out))
    bridge=pd.read_csv(plot/'level_policy_bridge_join_corrected36.csv',dtype={'animal_id':str})
    reference=pd.read_csv(plot/'level_policy_bridge_stats_corrected36.csv')
    rows=[]
    for ref in reference.itertuples():
        x,y=bridge[ref.x],bridge[ref.y]
        r,p=stats.pearsonr(x,y)
        fit=cr1((y-y.mean())/y.std(ddof=0),bridge.animal_id,(x-x.mean())/x.std(ddof=0))
        rows.append(dict(x=ref.x,y=ref.y,n_pairs=len(bridge),n_animals=bridge.animal_id.nunique(),pearson_r=r,
                         cluster_beta_standardized=fit['estimate'],cluster_se=fit['cluster_se'],
                         cluster_t=fit['t_value'],cluster_df=fit['df'],cluster_p=fit['p_two_sided']))
    actual=pd.DataFrame(rows)
    if set(actual.y) != {'Bf','Ba','Lf','Lo_depth_weighted'} or len(actual) != 4:
        raise AssertionError('ED10c BH family must contain exactly the four displayed fixed-30-minute policies')
    actual['cluster_bh4_adjusted_p']=multipletests(actual.cluster_p,method='fdr_bh')[1]
    results.append(compare_table(actual,reference,['x','y'],'ED10c_policy_animal_clustered_BH4',out))
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
