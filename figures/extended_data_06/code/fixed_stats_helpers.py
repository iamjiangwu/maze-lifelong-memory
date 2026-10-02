from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
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
    crit = float(stats.t.ppf(0.975, df))
    return {'n_pair_rows': len(work), 'n_animal_clusters': n_groups, 'estimate': estimate, 'cluster_se': se, 't_value': t_value, 'df': df, 'p_two_sided': p_two, 'ci_low': estimate - crit * se, 'ci_high': estimate + crit * se}


def _build_stats(pair_inclusion: pd.DataFrame) -> pd.DataFrame:
    retained = ((pair_inclusion.window_type.eq('maze_time') & pair_inclusion.window_size.isin([10.0, 20.0, 30.0])) | (pair_inclusion.window_type.eq('tile_change') & pair_inclusion.window_size.eq(3000.0)))
    if not retained.all():
        raise ValueError('Fixed-reference publication inputs contain an unshown window')
    rows: list[dict[str, object]] = []
    group_cols = ['metric', 'metric_key', 'metric_label', 'window_type', 'window_size', 'window_size_label']
    for (keys, data) in pair_inclusion.groupby(group_cols, sort=False):
        common = dict(zip(group_cols, keys))
        rise = _cluster_intercept_test(data[data['included_in_slope_test']], 'slope_per_hour_or_ktile')
        rows.append({**common, 'estimand': 'mean_pairwise_within_day1_slope', 'estimate_unit': 'r per hour' if common['window_type'] == 'maze_time' else 'r per 1,000 tile changes', **rise})
    stats_frame = pd.DataFrame(rows)
    if len(stats_frame) != 12:
        raise ValueError('Fixed-reference correction family must contain twelve tests')
    for (estimand, index) in stats_frame.groupby('estimand').groups.items():
        index = list(index)
        stats_frame.loc[index, 'q_bh12_two_sided'] = multipletests(stats_frame.loc[index, 'p_two_sided'].to_numpy(float), method='fdr_bh')[1]
    stats_frame['correction_family'] = 'BH across 12 displayed fixed-reference slope tests: 3 metrics x (10/20/30 min and 3000 tiles)'
    stats_frame['statistical_unit'] = 'same-stage Day1/Day2 session pair; all rows retained'
    stats_frame['dependence_handling'] = 'animal-clustered CR1 covariance; Student t reference with df=clusters-1'
    stats_frame['primary_reported_p'] = 'two-sided p; BH across the twelve displayed fixed-reference definitions'
    return stats_frame.sort_values(['estimand', 'window_type', 'metric_key', 'window_size']).reset_index(drop=True)
