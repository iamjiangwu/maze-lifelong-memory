"""Behavioral contrasts calculated from supplied session observations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

def read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip', **kwargs)

def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1)
    return out


def fit(y, design, clusters) -> dict:
    (y, design, clusters) = (np.asarray(y, float), np.asarray(design, float), np.asarray(clusters, str))
    keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
    (y, design, clusters) = (y[keep], design[keep], clusters[keep])
    model = sm.OLS(y, design).fit()
    covariance = cov_cluster(model, clusters, use_correction=True)
    g = len(np.unique(clusters))
    return {'beta': model.params, 'covariance': covariance, 'se': np.sqrt(np.maximum(np.diag(covariance), 0)), 'df': g - 1, 'n_rows': len(y), 'n_animals': g, 'r2': model.rsquared, 'ordinary': model}

def contrast(model, vector) -> dict:
    vector = np.asarray(vector, float)
    estimate = float(vector @ model['beta'])
    se = float(np.sqrt(max(vector @ model['covariance'] @ vector, 0)))
    t = estimate / se
    critical = stats.t.ppf(0.975, model['df'])
    return dict(estimate=estimate, se=se, t=t, df=model['df'], p=float(2 * stats.t.sf(abs(t), model['df'])), ci_low=estimate - critical * se, ci_high=estimate + critical * se, n_rows=model['n_rows'], n_animals=model['n_animals'])



def main1(root: Path, audit: StatisticsOutput):
    p = root / 'main_figure_1/source/frozen_source/plot_ready'
    for (prefix, group, effect, fold, family) in [('reward_acquisition_day1_month', 'month', 'coef_log_later_minus_naive', 'fold_change_later_over_naive', 'four naive-versus-return-month contrasts BH4'), ('long_bout_group', 'group_index', 'coef_log_group_b_minus_group_a', 'fold_change_group_b_over_group_a', 'three pairwise reward-context contrasts BH3')]:
        points = read(p / (prefix + '_points.csv'))
        saved = read(p / (prefix + '_clustered_contrasts.csv'))
        rows = []
        for row in saved.itertuples():
            sub = points.loc[points[group].isin([row.group_a, row.group_b])].copy()
            sub = sub.loc[np.isfinite(sub.lb0) & sub.lb0.gt(0)]
            indicator = sub[group].eq(row.group_b).to_numpy(float)
            model = fit(np.log(sub.lb0), np.column_stack([np.ones(len(sub)), indicator]), sub.animal_id)
            c = contrast(model, [0, 1])
            rows.append(dict(comparison=row.comparison, **{effect: c['estimate'], fold: np.exp(c['estimate'])}, n_sessions=c['n_rows'], n_animals=c['n_animals'], cluster_ols_t=c['t'], cluster_ols_p=c['p']))
        result = pd.DataFrame(rows)
        result['cluster_ols_q_bh'] = bh(result.cluster_ols_p)
        audit.save('main1_' + prefix, result, unit='session observations; OLS with standard errors clustered by animal; log(lb0) response', family=family)
    returns = read(p / 'reward_acquisition_day1_month_points.csv')
    returns = returns.loc[returns.month.isin([3, 6, 9, 12]) & returns.lb0.gt(0)].copy()
    c = contrast(fit(np.log(returns.lb0), np.column_stack([np.ones(len(returns)), returns.month / 3]), returns.animal_id), [0, 1])
    trend = pd.DataFrame([dict(analysis='within_return_experience_trend', n_rows=c['n_rows'], n_animals=c['n_animals'], estimate_log=c['estimate'], cluster_se=c['se'], t=c['t'], df=c['df'], p_two_sided_cluster_t=c['p'], fold_change_per_unit=np.exp(c['estimate']), ci95_log_low=c['ci_low'], ci95_log_high=c['ci_high'])])
    audit.save('main1_within_return_experience_trend', trend, unit='56 return sessions from 17 animals; log-time slope per return cycle; standard errors clustered by animal', family='one captioned trend test; nominal two-sided p')
    points = read(p / 'session_route.csv')
    saved = read(p / 'route_stats.csv')
    group_saved = read(p / 'route_group_stats.csv')
    groups = group_saved.condition.tolist()
    x = np.column_stack([points.condition.eq(g).to_numpy(float) for g in groups])
    model = fit(points.route_fraction_correct, x, points.animal)
    rows = []
    for row in saved.itertuples():
        cvector = np.zeros(len(groups))
        cvector[groups.index(row.target_condition)] = 1
        cvector[groups.index(row.baseline_condition)] = -1
        c = contrast(model, cvector)
        rows.append(dict(comparison=row.comparison, mean_diff_target_minus_baseline=c['estimate'], cluster_se=c['se'], cluster_ci95_low=c['ci_low'], cluster_ci95_high=c['ci_high'], t_cluster=c['t'], df_cluster=c['df'], p_cluster=c['p'], n_rows_total=c['n_rows'], n_clusters_total=c['n_animals']))
    result = pd.DataFrame(rows)
    result['q_bh6'] = bh(result.p_cluster)
    audit.save('main1_route_contrasts', result, unit='162 sessions; four-level saturated condition model; standard errors clustered by animal (33 animals)', family='six route-score pairwise contrasts BH6')
    rows = []
    for (i, g) in enumerate(groups):
        vector = np.zeros(len(groups))
        vector[i] = 1
        c = contrast(model, vector)
        sub = points.loc[points.condition.eq(g)]
        rows.append(dict(condition=g, mean_route_fraction_correct=c['estimate'], cluster_se=c['se'], cluster_ci95_low=c['ci_low'], cluster_ci95_high=c['ci_high'], n_sessions=len(sub), n_animals=sub.animal.nunique(), n_clusters_total=c['n_animals'], df_cluster=c['df']))
    result = pd.DataFrame(rows)
    audit.save('main1_route_group_ci', result, unit='condition means from same saturated CR1 model', family='none; 95% marginal t(G-1) intervals')
    for (prefix, key) in [('reward_acquisition_day1_month', 'month'), ('long_bout_group', 'group_index')]:
        points = read(p / (prefix + '_points.csv'))
        saved = read(p / (prefix + '_summary.csv'))
        rows = []
        for (g, sub) in points.groupby(key):
            x = pd.to_numeric(sub.lb0, errors='coerce').dropna().to_numpy(float)
            rows.append({key: g, 'mean': x.mean(), 'sem': stats.sem(x), 'std': x.std(ddof=1), 'median': np.median(x), 'min': x.min(), 'max': x.max(), 'n_sessions': len(x), 'n_animals': sub.animal_id.nunique()})
        result = pd.DataFrame(rows)
        audit.save('main1_' + prefix + '_summary', result, unit='session-level descriptive mean/SEM', family='none')



class StatisticsOutput:
    """Write calculated estimates and their statistical units."""
    def __init__(self, output: Path):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.analyses = []

    def save(self, name, values, *, unit, family):
        values.to_csv(self.output / (name + '.csv'), index=False)
        self.analyses.append({'analysis':name, 'rows':len(values), 'unit':unit, 'correction_family':family})
