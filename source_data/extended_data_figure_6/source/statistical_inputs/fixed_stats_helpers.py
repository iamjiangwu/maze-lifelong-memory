from __future__ import annotations
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster
def _cluster_intercept_test(data: pd.DataFrame, value_col: str) -> dict[str, object]:
    work = data[[value_col, 'animal']].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
    work = work.dropna()
    groups = work['animal'].astype(str)
    n_groups = groups.nunique()
    fit = sm.OLS(work[value_col].to_numpy(float), np.ones((len(work), 1))).fit(cov_type='cluster', cov_kwds={'groups': groups, 'use_correction': True})
    estimate = float(fit.params[0])
    se = float(fit.bse[0])
    df = n_groups - 1
    t_value = estimate / se
    p_two = float(2.0 * stats.t.sf(abs(t_value), df))
    p_directional = float(stats.t.sf(t_value, df))
    crit = float(stats.t.ppf(0.975, df))
    return {'n_pair_rows': len(work), 'n_animal_clusters': n_groups, 'estimate': estimate, 'cluster_se': se, 't_value': t_value, 'df': df, 'p_two_sided': p_two, 'p_directional_positive': p_directional, 'ci_low': estimate - crit * se, 'ci_high': estimate + crit * se}

def _cluster_experience_test(data: pd.DataFrame) -> dict[str, object]:
    work = data[['first_value', 'experience_index', 'animal']].copy()
    work['first_value'] = pd.to_numeric(work['first_value'], errors='coerce')
    work['experience_index'] = pd.to_numeric(work['experience_index'], errors='coerce')
    work = work.dropna()
    groups = work['animal'].astype(str)
    n_groups = groups.nunique()
    design = sm.add_constant(work[['experience_index']], has_constant='add')
    fit = sm.OLS(work['first_value'], design).fit(cov_type='cluster', cov_kwds={'groups': groups, 'use_correction': True})
    estimate = float(fit.params['experience_index'])
    se = float(fit.bse['experience_index'])
    df = n_groups - 1
    t_value = estimate / se
    p_two = float(2.0 * stats.t.sf(abs(t_value), df))
    p_directional = float(stats.t.sf(t_value, df))
    crit = float(stats.t.ppf(0.975, df))
    return {'n_pair_rows': len(work), 'n_animal_clusters': n_groups, 'estimate': estimate, 'cluster_se': se, 't_value': t_value, 'df': df, 'p_two_sided': p_two, 'p_directional_positive': p_directional, 'ci_low': estimate - crit * se, 'ci_high': estimate + crit * se}

def _build_stats(pair_inclusion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ['metric', 'metric_key', 'metric_label', 'window_type', 'window_size', 'window_size_label']
    for (keys, data) in pair_inclusion.groupby(group_cols, sort=False):
        common = dict(zip(group_cols, keys))
        rise = _cluster_intercept_test(data[data['included_in_slope_test']], 'slope_per_hour_or_ktile')
        rows.append({**common, 'estimand': 'mean_pairwise_within_day1_slope', 'estimate_unit': 'r per hour' if common['window_type'] == 'maze_time' else 'r per 1,000 tile changes', **rise})
        upward = _cluster_experience_test(data[data['included_in_experience_test']])
        rows.append({**common, 'estimand': 'marginal_first_window_shift_per_experience', 'estimate_unit': 'r per experience step', **upward})
    stats_frame = pd.DataFrame(rows)
    for (estimand, index) in stats_frame.groupby('estimand').groups.items():
        index = list(index)
        stats_frame.loc[index, 'q_bh18_two_sided'] = multipletests(stats_frame.loc[index, 'p_two_sided'].to_numpy(float), method='fdr_bh')[1]
        stats_frame.loc[index, 'q_bh18_directional_positive'] = multipletests(stats_frame.loc[index, 'p_directional_positive'].to_numpy(float), method='fdr_bh')[1]
    stats_frame['correction_family'] = 'BH within estimand across 18 planned tests: 3 metrics x 3 spans x 2 progress axes'
    stats_frame['statistical_unit'] = 'same-stage Day1/Day2 session pair; all rows retained'
    stats_frame['dependence_handling'] = 'animal-clustered CR1 covariance; Student t reference with df=clusters-1'
    stats_frame['primary_reported_p'] = 'two-sided p and BH q; directional positive p/q retained as prespecified sensitivity'
    return stats_frame.sort_values(['estimand', 'window_type', 'metric_key', 'window_size']).reset_index(drop=True)
