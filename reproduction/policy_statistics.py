"""Turning policy and route contrasts from the included observations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p, kind='stable')
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return q

def cr1(y, x, animals, weights=None):
    (y, x, animals) = (np.asarray(y, float), np.asarray(x, float), np.asarray(animals, str))
    if x.ndim == 1:
        x = x[:, None]
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(weights) & (weights > 0)
    (y, x, animals, weights) = (y[keep], x[keep], animals[keep], weights[keep])
    (n, k) = x.shape
    unique = np.unique(animals)
    g = len(unique)
    if n <= k or g < 2:
        raise ValueError(f'Insufficient CR1 support: N={n}, K={k}, G={g}')
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((k, k))
    for animal in unique:
        mask = animals == animal
        score = x[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    covariance = g / (g - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), g - 1)
    radius = stats.t.ppf(0.975, g - 1) * se
    total = np.sum((y - y.mean()) ** 2)
    return dict(beta=beta, se=se, t=t, p=p, low=beta - radius, high=beta + radius, covariance=covariance, n=n, g=g, df=g - 1, r2=1 - np.sum(residual ** 2) / total if total else np.nan)

def ed2(source, output, analyses):
    policy = pd.read_csv(source / 'analysis_inputs/natural/fixed_time_policy_rows.csv', dtype={'animal_id': str})
    if not policy.window_minutes.eq(30).all():
        raise AssertionError('ED2 publication inputs must contain only first-30-minute policies')
    for (family, savedname, metrics, gate, qsize) in (('policy', 'equal_junction_policy_group_cluster_stats.csv', ['Bf', 'Ba', 'Lf', 'Lo'], 'primary_policy_support', 4), ('depth', 'equal_junction_depth_group_cluster_stats.csv', ['Lo_shallow', 'Lo_mid', 'Lo_deep'], 'primary_depth_support', 3)):
        saved = pd.read_csv(source / 'analysis_inputs/natural' / savedname)
        if not saved.window_minutes.eq(30).all():
            raise AssertionError('ED2 publication tests must use the displayed 30-minute window')
        if family == 'depth' and (not saved.support_policy.eq('all_depth_strata_n_ge_10').all()):
            raise AssertionError('ED2 depth tests must use the displayed all-strata support')
        rows = []
        for row in saved.to_dict('records'):
            sub = policy.loc[policy[gate].astype(bool) & policy.group.isin([row['target'], row['baseline']])]
            f = cr1(sub[row['metric']], np.column_stack([np.ones(len(sub)), sub.group.eq(row['target'])]), sub.animal_id)
            rows.append({**{k: row[k] for k in (['support_policy'] if family == 'depth' else []) + ['window_minutes', 'contrast_id', 'metric']}, 'beta_target_minus_baseline': f['beta'][1], 'cluster_se': f['se'][1], 'ci_low': f['low'][1], 'ci_high': f['high'][1], 't_cluster': f['t'][1], 'p_cluster': f['p'][1], 'n_target_rows': int(sub.group.eq(row['target']).sum()), 'n_baseline_rows': int(sub.group.eq(row['baseline']).sum()), 'n_mice_union': f['g']})
        actual = pd.DataFrame(rows)
        scope = ['support_policy'] if family == 'depth' else []
        suffix = f'q_bh{qsize}_within_window_contrast'
        actual[suffix] = np.nan
        for ids in actual.groupby(scope + ['window_minutes', 'contrast_id']).groups.values():
            if len(ids) != qsize:
                raise AssertionError('ED2 publication correction family is incomplete')
            actual.loc[ids, suffix] = bh(actual.loc[ids, 'p_cluster'])
        save_statistics('ed2_' + family, actual, output, analyses)
    simulation = pd.read_csv(source / 'statistical_inputs/matched_counterfactual_session_rows.csv', dtype={'animal_id': str})
    parameters = ('Bf', 'Ba', 'Lf', 'Lo')
    common = simulation.loc[simulation.pair_common_primary_eligible.astype(bool)]
    four = simulation.loc[simulation.pair_four_primary_eligible.astype(bool)]

    def pools(frame, columns):
        grouped = frame.groupby('animal_id', sort=True)[columns]
        return (grouped.sum().to_numpy(float), grouped.size().to_numpy(float))
    cols = ['observed_paired_difference', 'four_simulated_paired_difference', 'depth_simulated_paired_difference']
    (cs, cn) = pools(common, cols)
    (fs, fn) = pools(four, ['contribution_' + p for p in parameters])
    rng = np.random.default_rng(20260820 + 100000)
    gapdraw = []
    condraw = []
    for _ in range(2000):
        ids = rng.choice(len(cn), size=len(cn), replace=True)
        v = cs[ids].sum(axis=0) / cn[ids].sum()
        gapdraw.append([*v, v[1] / v[0], v[2] / v[0]])
        ids = rng.choice(len(fn), size=len(fn), replace=True)
        condraw.append(fs[ids].sum(axis=0) / fn[ids].sum())
    gapdraw = np.asarray(gapdraw)
    condraw = np.asarray(condraw)
    contributions = pd.DataFrame([dict(parameter=p, delta_efficiency_mean=four['contribution_' + p].mean(), delta_efficiency_ci_low=np.percentile(condraw[:, j], 2.5), delta_efficiency_ci_high=np.percentile(condraw[:, j], 97.5)) for (j, p) in enumerate(parameters)])
    save_statistics('ed2_matched_policy_contributions', contributions, output, analyses)
    point = common[cols].mean().to_numpy(float)
    gap = pd.DataFrame([dict(model=m, delta_real=point[0], delta_simulated=point[j], fraction_recapitulated=point[j] / point[0], fraction_ci_low=np.percentile(gapdraw[:, j + 2], 2.5), fraction_ci_high=np.percentile(gapdraw[:, j + 2], 97.5), delta_simulated_ci_low=np.percentile(gapdraw[:, j], 2.5), delta_simulated_ci_high=np.percentile(gapdraw[:, j], 97.5)) for (j, m) in [(1, 'four_bias'), (2, 'depth_stratified')]])
    save_statistics('ed2_matched_gap', gap, output, analyses)

def ed3(source, output, analyses):
    rows = pd.read_csv(source / 'plot_ready/session_route.csv', dtype={'animal': str})
    saved = pd.read_csv(source / 'plot_ready/route_stats.csv')
    groups = ('naive', 'return', 'experienced_day2', 'latent_day2')
    f = cr1(rows.route_fraction_correct, np.column_stack([rows.condition.eq(g) for g in groups]), rows.animal)
    contrasts = []
    for row in saved.to_dict('records'):
        c = np.zeros(4)
        c[groups.index(row['target_condition'])] = 1
        c[groups.index(row['baseline_condition'])] = -1
        estimate = c @ f['beta']
        se = np.sqrt(c @ f['covariance'] @ c)
        t = estimate / se
        radius = stats.t.ppf(0.975, f['df']) * se
        contrasts.append(dict(comparison=row['comparison'], mean_diff_target_minus_baseline=estimate, cluster_se=se, cluster_ci95_low=estimate - radius, cluster_ci95_high=estimate + radius, p_cluster=2 * stats.t.sf(abs(t), f['df']), n_rows_total=f['n'], n_clusters_total=f['g']))
    actual = pd.DataFrame(contrasts)
    actual['q_bh6'] = bh(actual.p_cluster)
    save_statistics('ed3_route_bh6', actual, output, analyses)
    source_rows = pd.read_csv(source / 'plot_ready/fluency_plot_rows.csv', dtype={'animal': str})
    saved = pd.read_csv(source / 'plot_ready/fluency_statistics.csv')
    records = []
    saved = saved.loc[saved.scale.eq('raw_seconds')]
    for row in saved.to_dict('records'):
        sub = source_rows.loc[source_rows.claim_id.eq(row['claim_id'])].copy()
        y = sub.mean_decision_time_s.to_numpy(float)
        x = np.column_stack([np.ones(len(sub)), sub.condition.eq('return')])
        f = cr1(y, x, sub.animal)
        records.append(dict(claim_id=row['claim_id'], scale=row['scale'], estimate_return_minus_latent=f['beta'][1], cluster_se=f['se'][1], ci95_low=f['low'][1], ci95_high=f['high'][1], p_cluster_two_sided=f['p'][1], n_rows=f['n'], n_mice=f['g']))
    actual = pd.DataFrame(records)
    for ids in actual.groupby('scale').groups.values():
        actual.loc[ids, 'q_bh4_displayed_story'] = bh(actual.loc[ids, 'p_cluster_two_sided'])
    save_statistics('ed3_fluency_bh4', actual, output, analyses)


def save_statistics(name, values, output, analyses):
    """Write a recalculated statistic table."""
    values.to_csv(Path(output) / (name + '.csv'), index=False)
    analyses.append({'analysis':name, 'rows':len(values)})
