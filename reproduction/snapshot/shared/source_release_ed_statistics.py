"""Recreate ED2/3/5/7/9 inference from frozen derived observation rows.

This deliberately does not load acquisition files or use stored p values as
inputs. Stored inference tables are comparison targets only. CR1 uses the
maintained finite-sample factor G/(G-1)*(N-1)/(N-K) and Student t(G-1).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

CORRELATIONS = ("place_map_correlation", "pv_correlation", "level_map_correlation", "ensemble_rate_correlation")


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p, kind="stable")
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return q


def cr1(y, x, animals, weights=None):
    # Rows are observations; x columns are model coefficients. Repeated rows
    # from one animal share a covariance cluster. Positive weights affect the fit.
    y, x, animals = np.asarray(y, float), np.asarray(x, float), np.asarray(animals, str)
    if x.ndim == 1:
        x = x[:, None]
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(weights) & (weights > 0)
    y, x, animals, weights = y[keep], x[keep], animals[keep], weights[keep]
    n, k = x.shape
    unique = np.unique(animals)
    g = len(unique)
    if n <= k or g < 2:
        raise ValueError(f"Insufficient CR1 support: N={n}, K={k}, G={g}")
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((k, k))
    for animal in unique:
        mask = animals == animal
        # Sum residual-weighted design rows within animal before forming the
        # covariance, retaining dependence among that animal's observations.
        score = x[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    # CR1 finite-sample factor; inference below uses G - 1 animal degrees of freedom.
    covariance = g / (g - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), g - 1)
    radius = stats.t.ppf(.975, g - 1) * se
    total = np.sum((y - y.mean()) ** 2)
    return dict(beta=beta, se=se, t=t, p=p, low=beta-radius, high=beta+radius,
                covariance=covariance, n=n, g=g, df=g-1,
                r2=1-np.sum(residual**2)/total if total else np.nan)


def intercept(y, animals):
    f = cr1(y, np.ones((len(y), 1)), animals)
    return dict(estimate=f["beta"][0], se_cr1=f["se"][0], ci95_low=f["low"][0],
                ci95_high=f["high"][0], t=f["t"][0], cluster_df=f["df"],
                p_two_sided=f["p"][0], p_greater=stats.t.sf(f["t"][0], f["df"]),
                n_pairs=f["n"], n_animals=f["g"])


def slope(frame, xcolumn, ycolumn, animalcolumn):
    sub = frame.dropna(subset=[xcolumn, ycolumn, animalcolumn])
    x, y = sub[xcolumn].to_numpy(float), sub[ycolumn].to_numpy(float)
    f = cr1(y, np.column_stack([np.ones(len(x)), x]), sub[animalcolumn])
    return dict(intercept=f["beta"][0], slope=f["beta"][1], slope_se=f["se"][1],
                slope_t=f["t"][1], p_two_sided=f["p"][1], df=f["df"],
                n_rows=f["n"], n_animals=f["g"], pearson_r=np.corrcoef(x, y)[0,1], r2=f["r2"])


def compare(name, actual, target, keys, columns, output, analyses):
    if not columns:
        raise AssertionError(f"{name}: empty numeric comparison contract")
    for side, table in [("regenerated", actual), ("reference", target)]:
        missing = set(keys).union(columns).difference(table.columns)
        if missing or table.empty:
            raise AssertionError(f"{name}: {side} is empty or missing required columns: {sorted(missing)}")
    actual.to_csv(output / (name + ".csv"), index=False)
    left = actual.set_index(keys).sort_index()
    right = target.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise AssertionError(f"{name}: observation/test identifiers disagree")
    checks = []
    for column in columns:
        a, b = pd.to_numeric(left[column]).to_numpy(float), pd.to_numeric(right[column]).to_numpy(float)
        probability = bool({"p", "q"}.intersection(column.lower().split("_")))
        # Preserve relative accuracy for tiny raw and adjusted probabilities.
        atol = 1e-300 if probability else 1e-10
        same_nan = np.isnan(a) & np.isnan(b)
        ok = same_nan | (np.isfinite(a) & np.isfinite(b) & (np.abs(a-b) <= atol + 1e-8*np.abs(b)))
        delta = np.abs(a-b)
        checks.append(dict(column=column, passed=bool(ok.all()), max_absolute_difference=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0))
    report = dict(analysis=name, status="PASS" if all(c["passed"] for c in checks) else "FAIL",
                  regenerated_rows=len(actual), source_unit="derived observation rows; no saved p-value inputs", checks=checks)
    analyses.append(report)


def ed2(source, output, analyses):
    policy = pd.read_csv(source / "analysis_inputs/natural/fixed_time_policy_rows.csv", dtype={"animal_id": str})
    if not policy.window_minutes.eq(30).all():
        raise AssertionError("ED2 publication inputs must contain only first-30-minute policies")
    for family, savedname, metrics, gate, qsize in (
        ("policy", "equal_junction_policy_group_cluster_stats.csv", ["Bf", "Ba", "Lf", "Lo"], "primary_policy_support", 4),
        ("depth", "equal_junction_depth_group_cluster_stats.csv", ["Lo_shallow", "Lo_mid", "Lo_deep"], "primary_depth_support", 3),
    ):
        saved = pd.read_csv(source / "analysis_inputs/natural" / savedname)
        if not saved.window_minutes.eq(30).all():
            raise AssertionError("ED2 publication tests must use the displayed 30-minute window")
        if family == "depth" and not saved.support_policy.eq("all_depth_strata_n_ge_10").all():
            raise AssertionError("ED2 depth tests must use the displayed all-strata support")
        rows=[]
        for row in saved.to_dict("records"):
            sub = policy.loc[policy[gate].astype(bool) & policy.group.isin([row["target"], row["baseline"]])]
            f = cr1(sub[row["metric"]], np.column_stack([np.ones(len(sub)), sub.group.eq(row["target"])]), sub.animal_id)
            rows.append({**{k: row[k] for k in (["support_policy"] if family == "depth" else []) + ["window_minutes", "contrast_id", "metric"]},
                         "beta_target_minus_baseline":f["beta"][1], "cluster_se":f["se"][1], "ci_low":f["low"][1], "ci_high":f["high"][1], "t_cluster":f["t"][1], "p_cluster":f["p"][1],
                         "n_target_rows":int(sub.group.eq(row["target"]).sum()), "n_baseline_rows":int(sub.group.eq(row["baseline"]).sum()), "n_mice_union":f["g"]})
        actual=pd.DataFrame(rows)
        scope=["support_policy"] if family == "depth" else []
        suffix = f"q_bh{qsize}_within_window_contrast"
        actual[suffix]=np.nan
        for ids in actual.groupby(scope+["window_minutes","contrast_id"]).groups.values():
            if len(ids) != qsize:
                raise AssertionError("ED2 publication correction family is incomplete")
            actual.loc[ids,suffix]=bh(actual.loc[ids,"p_cluster"])
        compare("ed2_"+family,actual,saved,scope+["window_minutes","contrast_id","metric"],
                ["beta_target_minus_baseline","cluster_se","ci_low","ci_high","p_cluster",suffix,"n_target_rows","n_baseline_rows","n_mice_union"],output,analyses)
    # Matched-policy counterfactual effects were already simulated; only their
    # animal-cluster bootstrap is repeated here, with the original seeded draws.
    simulation=pd.read_csv(source/"statistical_inputs/matched_counterfactual_session_rows.csv",dtype={"animal_id":str})
    parameters=("Bf","Ba","Lf","Lo")
    common=simulation.loc[simulation.pair_common_primary_eligible.astype(bool)]
    four=simulation.loc[simulation.pair_four_primary_eligible.astype(bool)]
    def pools(frame, columns):
        grouped=frame.groupby("animal_id",sort=True)[columns]
        return grouped.sum().to_numpy(float),grouped.size().to_numpy(float)
    cols=["observed_paired_difference","four_simulated_paired_difference","depth_simulated_paired_difference"]
    cs,cn=pools(common,cols); fs,fn=pools(four,["contribution_"+p for p in parameters])
    rng=np.random.default_rng(20260820+100000); gapdraw=[]; condraw=[]
    for _ in range(2000):
        ids=rng.choice(len(cn),size=len(cn),replace=True); v=cs[ids].sum(axis=0)/cn[ids].sum(); gapdraw.append([*v,v[1]/v[0],v[2]/v[0]])
        ids=rng.choice(len(fn),size=len(fn),replace=True); condraw.append(fs[ids].sum(axis=0)/fn[ids].sum())
    gapdraw=np.asarray(gapdraw); condraw=np.asarray(condraw)
    contributions=pd.DataFrame([dict(parameter=p,delta_efficiency_mean=four["contribution_"+p].mean(),delta_efficiency_ci_low=np.percentile(condraw[:,j],2.5),delta_efficiency_ci_high=np.percentile(condraw[:,j],97.5)) for j,p in enumerate(parameters)])
    compare("ed2_matched_policy_contributions",contributions,pd.read_csv(source/"plot_ready/panel_d_policy_contributions.csv"),["parameter"],["delta_efficiency_mean","delta_efficiency_ci_low","delta_efficiency_ci_high"],output,analyses)
    point=common[cols].mean().to_numpy(float)
    gap=pd.DataFrame([dict(model=m,delta_real=point[0],delta_simulated=point[j],fraction_recapitulated=point[j]/point[0],fraction_ci_low=np.percentile(gapdraw[:,j+2],2.5),fraction_ci_high=np.percentile(gapdraw[:,j+2],97.5),delta_simulated_ci_low=np.percentile(gapdraw[:,j],2.5),delta_simulated_ci_high=np.percentile(gapdraw[:,j],97.5)) for j,m in [(1,"four_bias"),(2,"depth_stratified")]])
    compare("ed2_matched_gap",gap,pd.read_csv(source/"plot_ready/panel_h_four_vs_depth_gap.csv"),["model"],["delta_real","delta_simulated","fraction_recapitulated","fraction_ci_low","fraction_ci_high"],output,analyses)


def ed3(source, output, analyses):
    rows=pd.read_csv(source/"plot_ready/ed04_session_route_current_plus_new12.csv",dtype={"animal":str})
    saved=pd.read_csv(source/"plot_ready/ed04_route_stats_current_plus_new12.csv")
    groups=("naive","return","experienced_day2","latent_day2")
    f=cr1(rows.route_fraction_correct,np.column_stack([rows.condition.eq(g) for g in groups]),rows.animal)
    contrasts=[]
    for row in saved.to_dict("records"):
        c=np.zeros(4);c[groups.index(row["target_condition"])]=1;c[groups.index(row["baseline_condition"])]=-1
        estimate=c@f["beta"];se=np.sqrt(c@f["covariance"]@c);t=estimate/se;radius=stats.t.ppf(.975,f["df"])*se
        contrasts.append(dict(comparison=row["comparison"],mean_diff_target_minus_baseline=estimate,cluster_se=se,cluster_ci95_low=estimate-radius,cluster_ci95_high=estimate+radius,p_cluster=2*stats.t.sf(abs(t),f["df"]),n_rows_total=f["n"],n_clusters_total=f["g"]))
    actual=pd.DataFrame(contrasts);actual["q_bh6"]=bh(actual.p_cluster)
    compare("ed3_route_bh6",actual,saved,["comparison"],["mean_diff_target_minus_baseline","cluster_se","cluster_ci95_low","cluster_ci95_high","p_cluster","q_bh6","n_rows_total","n_clusters_total"],output,analyses)
    source_rows=pd.read_csv(source/"plot_ready/fluency_plot_rows.csv",dtype={"animal":str})
    saved=pd.read_csv(source/"plot_ready/fluency_statistics.csv");records=[]
    saved=saved.loc[saved.scale.eq("raw_seconds")]
    for row in saved.to_dict("records"):
        sub=source_rows.loc[source_rows.claim_id.eq(row["claim_id"])].copy()
        y=sub.mean_decision_time_s.to_numpy(float)
        x=np.column_stack([np.ones(len(sub)),sub.condition.eq("return")]);f=cr1(y,x,sub.animal)
        records.append(dict(claim_id=row["claim_id"],scale=row["scale"],estimate_return_minus_latent=f["beta"][1],cluster_se=f["se"][1],ci95_low=f["low"][1],ci95_high=f["high"][1],p_cluster_two_sided=f["p"][1],n_rows=f["n"],n_mice=f["g"]))
    actual=pd.DataFrame(records)
    for ids in actual.groupby("scale").groups.values():
        actual.loc[ids,"q_bh4_displayed_story"]=bh(actual.loc[ids,"p_cluster_two_sided"])
    compare("ed3_fluency_bh4",actual,saved,["claim_id","scale"],["estimate_return_minus_latent","cluster_se","ci95_low","ci95_high","p_cluster_two_sided","q_bh4_displayed_story","n_rows","n_mice"],output,analyses)


def ed5(source, output, analyses):
    # Exact maintained producer: loading this script does not execute its CLI.
    import importlib.util
    spec=importlib.util.spec_from_file_location("ed5_frozen_statistics",source/"source_code/recompute_ed05_pair_weighted_day_combinations.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values,_=module.load_pair_values(source/"plot_ready")
    for name,actual,filename,keys,columns in [
        ("ed5_day_combination_means",module.summarize(values),"ed05_day_combination_session_pair_descriptive_cr1.csv",["metric","variant"],["mean","se_cr1","ci95_low_cr1","ci95_high_cr1","n_session_pairs","n_animals"]),
        ("ed5_day_combination_contrasts",module.contrasts(values),"ed05_day_combination_session_pair_contrasts_cr1_bh3.csv",["metric","contrast"],["mean_difference","se_cr1","ci95_low","ci95_high","p_value","p_exact_cluster_signflip","p_bh3_within_metric_cr1","p_bh3_within_metric_exact_cluster_signflip"]),
    ]: compare(name,actual,pd.read_csv(source/"plot_ready"/filename),keys,columns,output,analyses)
    pairs=pd.read_csv(source/"plot_ready/ed05e_context_only_all_outside_pair_rows.csv",dtype={"animal_id":str})
    saved=pd.read_csv(source/"plot_ready/ed05e_context_only_all_outside_clustered_stats.csv");records=[]
    if not pairs.variant.eq("d1v1").all() or not saved.variant.eq("d1v1").all():
        raise AssertionError("ED5e publication context inputs must contain only the displayed d1v1 comparison")
    for row in saved.to_dict("records"):
        sub=pairs.loc[pairs.variant.eq(row["variant"])]; f=intercept(sub["equal_time_"+row["analysis"]+"_r_mean"] if row["analysis"] in ["maze","outside"] else sub.equal_time_maze_minus_outside_mean,sub.animal_id)
        records.append(dict(variant=row["variant"],analysis=row["analysis"],beta=f["estimate"],cluster_se=f["se_cr1"],ci95_low=f["ci95_low"],ci95_high=f["ci95_high"],pvalue=f["p_two_sided"] if row["analysis"] == "maze_minus_outside" else np.nan,n_pair_rows=f["n_pairs"],n_animals=f["n_animals"]))
    actual=pd.DataFrame(records)
    compare("ed5_context_control",actual,saved,["variant","analysis"],["beta","cluster_se","ci95_low","ci95_high","pvalue","n_pair_rows","n_animals"],output,analyses)


def age_models(rows, agecolumn, stagecolumn, animalcolumn):
    results=[]
    for metric,sub in rows.groupby("metric",sort=False):
        sub=sub.dropna(subset=[agecolumn,stagecolumn,"value",animalcolumn])
        age=(sub[agecolumn].to_numpy(float)-sub[agecolumn].mean())/30
        stages=sorted(sub[stagecolumn].unique())
        x=np.column_stack([np.ones(len(sub)),age,*[sub[stagecolumn].eq(stage).to_numpy(float) for stage in stages[1:]]])
        f=cr1(sub.value,x,sub[animalcolumn]);results.append(dict(metric=metric,beta_per_30d=f["beta"][1],cluster_se_cr1=f["se"][1],t_cluster=f["t"][1],p_cluster_two_sided=f["p"][1],ci95_low=f["low"][1],ci95_high=f["high"][1],r_squared=f["r2"],n_rows=f["n"],n_mice=f["g"]))
    result=pd.DataFrame(results);result["p_bh"]=bh(result.p_cluster_two_sided);return result


def ed7(source,output,analyses):
    actual=age_models(pd.read_csv(source/"plot_ready/ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",dtype={"animal_id":str}),"age_day1_days","experience_index","animal_id")
    actual=actual.rename(columns={"p_bh":"q_bh3_displayed"})
    compare("ed7_first30_age_bh3",actual,pd.read_csv(source/"plot_ready/ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv"),["metric"],["beta_per_30d","cluster_se_cr1","ci95_low","ci95_high","p_cluster_two_sided","q_bh3_displayed","n_rows","n_mice"],output,analyses)
    actual=age_models(pd.read_csv(source/"plot_ready/ed09_long_gap_age_joined_corrected36.csv",dtype={"animal_id":str}),"age_dst_days","experience","animal_id")
    actual=actual.rename(columns={"cluster_se_cr1":"cluster_se","p_cluster_two_sided":"p_cluster","ci95_low":"ci_low","ci95_high":"ci_high","p_bh":"p_cluster_q_bh4","n_mice":"n_animals"})
    compare("ed7_long_gap_age_bh4",actual,pd.read_csv(source/"plot_ready/ed09_long_gap_age_transition_fixed_models_corrected36.csv"),["metric"],["beta_per_30d","cluster_se","ci_low","ci_high","p_cluster","p_cluster_q_bh4","n_rows","n_animals"],output,analyses)


def ed9(release,source,output,analyses):
    frozen=source/"source_data/frozen"
    # Panel f retains the current 36 recording-pair observations rather than
    # inferring tests from its pooled shifted-neuron curve. The two planned
    # effects use equal pair weight, CR1 by animal, t(G-1) and one-sided BH2.
    preferred=pd.read_csv(source/"statistical_inputs/preferred_level_pair_observations.csv",dtype={"animal_id":str,"pair_key":str})
    measures=["observed_adjacent_fraction","null_adjacent_fraction_mean","observed_mean_shift_distance","null_mean_shift_distance"]
    if not preferred.pair_key.is_unique or not np.isfinite(preferred[measures].to_numpy(float)).all():
        raise ValueError("Invalid preferred-level recording-pair observations")
    shifted_curve=pd.read_csv(frozen/"shift.csv")
    if (preferred.n_shifted.le(0).any() or preferred.n_shifted.gt(preferred.n_included_preference).any()
            or int(preferred.n_shifted.sum()) != int(shifted_curve.observed_count.sum())):
        raise ValueError("Preferred-level inference and displayed shifted-neuron curve have different support")
    preferred_actual=[]
    for metric,observed,null,direction,definition in [
        ("adjacent_shift_enrichment","observed_adjacent_fraction","null_adjacent_fraction_mean",1,"observed adjacent-shift fraction minus within-pair destination-ID null"),
        ("shorter_shift_distance","observed_mean_shift_distance","null_mean_shift_distance",-1,"null mean shift distance minus observed mean shift distance"),
    ]:
        fitted=intercept(direction*(preferred[observed]-preferred[null]),preferred.animal_id)
        preferred_actual.append(dict(metric=metric,effect_definition=definition,
            **{key:fitted[key] for key in ["estimate","se_cr1","ci95_low","ci95_high","t","cluster_df","n_pairs"]},
            p_directional=fitted["p_greater"],n_mice=fitted["n_animals"]))
    preferred_actual=pd.DataFrame(preferred_actual)
    preferred_actual["q_bh_two_preferred_level_metrics"]=bh(preferred_actual.p_directional)
    compare("ed9_preferred_level_bh2",preferred_actual,pd.read_csv(source/"statistical_inputs/preferred_level_statistics.csv"),["metric"],
        ["estimate","se_cr1","ci95_low","ci95_high","t","cluster_df","p_directional","q_bh_two_preferred_level_metrics","n_pairs","n_mice"],output,analyses)
    # Each retained recording pair has equal weight; these are descriptive
    # Student-t(N-1) intervals, not the clustered regression intervals below.
    gaprows=pd.read_csv(source/"statistical_inputs/level_gap_pair_values.csv",dtype={"animal":str})
    gap=[]
    for days,sub in gaprows.groupby("time_days",sort=True):
        observed=sub.observed_mean.to_numpy(float);mean=observed.mean()
        radius=stats.t.ppf(.975,len(sub)-1)*stats.sem(observed)
        gap.append(dict(time_days=days,observed_mean=mean,observed_ci95_low=mean-radius,observed_ci95_high=mean+radius,shuffle_mean=sub.shuffle_mean.mean(),n_pairs=len(sub),n_animals=sub.animal.nunique()))
    compare("ed9_level_gap_pair_means",pd.DataFrame(gap),pd.read_csv(frozen/"gap.csv"),["time_days"],["observed_mean","observed_ci95_low","observed_ci95_high","shuffle_mean","n_pairs","n_animals"],output,analyses)
    rows=pd.read_csv(frozen/"figure5_decoder_points.csv",dtype={"animal":str})
    wide=rows.pivot(index=["pair_id","animal"],columns="group",values="decoding_accuracy")
    value=intercept(wide["Data"]-wide["Shuffle"],wide.index.get_level_values("animal"))
    saved=json.loads((frozen/"figure5_decoder_stats.json").read_text())["cluster_ols"]
    actual=pd.DataFrame([dict(contrast="data_minus_shuffle",estimate=value["estimate"],se=value["se_cr1"],p_greater=value["p_greater"],p_two_sided=value["p_two_sided"])])
    compare("ed9_decoder",actual,pd.DataFrame([dict(contrast="data_minus_shuffle",**saved)]),["contrast"],["estimate","se","p_greater","p_two_sided"],output,analyses)
    pairs=pd.read_csv(source/"source_data/level_comparison/paired_contrasts.csv",dtype={"animal_id":str});actual=[]
    for suffix in ["horizontal","vertical"]:
        f=intercept(pairs["delta_level_minus_"+suffix],pairs.animal_id)
        actual.append(dict(contrast="level_minus_"+suffix+"8",**f,p_primary=f["p_greater"]))
    actual=pd.DataFrame(actual);actual["p_adjusted_bh2"]=bh(actual.p_primary)
    compare("ed9_axis_controls_bh2",actual,pd.read_csv(source/"source_data/level_comparison/statistics.csv"),["contrast"],["estimate","se_cr1","ci95_low","ci95_high","p_primary","p_adjusted_bh2"],output,analyses)
    rows=pd.read_csv(frozen/"within_level_pairs.csv",dtype={"animal_id":str});actual=[]
    for (scope,sub) in rows.loc[rows.scope.eq("global")].groupby("scope"):
        actual.append(dict(scope=scope,metric="raw",**intercept(sub["observed_minus_null_raw"],sub.animal_id)))
    actual=pd.DataFrame(actual).rename(columns={"p_greater":"p_directional","n_animals":"n_mice"})
    saved=pd.read_csv(frozen/"within_level_stats.csv").query("scope == 'global' and metric == 'raw'")
    compare("ed9_within_level_control",actual,saved,["scope","metric"],["estimate","se_cr1","ci95_low","ci95_high","p_directional","p_two_sided","n_pairs","n_mice"],output,analyses)
    main3=release/"main_figure_3/source/frozen_source/source_data"
    for name,points,stored,x in [("first30",main3/"first30/combined_first30_points_all_metrics.csv",source/"source_data/first30_addition/first30_stats.csv","experience_index"),("cycles",main3/"long_gap/main3_experience_trend_points_no_manual_mask.csv",frozen/"cycles_stats.csv","experience")]:
        allrows=pd.read_csv(points,dtype={"animal":str}); family=[]
        for metric in CORRELATIONS:
            family.append(dict(metric=metric,**slope(allrows.loc[allrows.metric.eq(metric)],x,"value","animal")))
        actual=pd.DataFrame(family);actual["q_bh_two_sided"]=bh(actual.p_two_sided);actual.to_csv(output/("ed9_"+name+"_entire_bh4_family.csv"),index=False)
        compare("ed9_"+name+"_bh4",actual.loc[actual.metric.eq("level_map_correlation")],pd.read_csv(stored),["metric"],["intercept","slope","slope_se","p_two_sided","q_bh_two_sided","pearson_r","n_rows","n_animals"],output,analyses)
    # ED9's current BH4 family combines three dense representation fits with
    # canonical (ungated) ensemble-rate session-pair inference.
    main2=release/"main_figure_2/source/source_data"
    rows=pd.read_csv(main2/"consecutive_gap/consecutive_gap_clean_long_gap_corrected_points.csv",dtype={"animal_id":str});family=[]
    for metric in CORRELATIONS[:3]:family.append(dict(metric=metric,**slope(rows.loc[rows.metric.eq(metric)],"prev_gap_correlation","next_gap_correlation","animal_id")))
    rate=pd.read_csv(main2/"rate_chain/session_pair_rate_consecutive_gap_points_canonical.csv",dtype={"animal_id":str})
    family.append(dict(metric="ensemble_rate_correlation",**slope(rate,"value_sorted_ab","value_sorted_bc","animal_id")))
    actual=pd.DataFrame(family);actual["q_bh_two_sided"]=bh(actual.p_two_sided);actual.to_csv(output/"ed9_consecutive_entire_bh4_family.csv",index=False)
    compare("ed9_consecutive_bh4",actual.loc[actual.metric.eq("level_map_correlation")],pd.read_csv(frozen/"consecutive_stats.csv"),["metric"],["intercept","slope","slope_se","p_two_sided","q_bh_two_sided","pearson_r","n_rows","n_animals"],output,analyses)
    # Empirical shuffle inference is recalculated from the recorded null draws.
    shuffle=source/"source_data/population_vector_shuffle"
    pairs=pd.read_csv(shuffle/"pair_summary.csv");null=pd.read_csv(shuffle/"null_draws.csv").mean_correlation.to_numpy(float)
    observed=pairs.observed_mean_correlation.mean(); summary=json.loads((shuffle/"summary.json").read_text())
    actual=pd.DataFrame([dict(test="null_distribution",observed_mean=observed,null_mean=null.mean(),null_sd=null.std(ddof=1),null_quantile025=np.quantile(null,.025),null_quantile975=np.quantile(null,.975),empirical_p_greater=(1+np.sum(null>=observed))/(len(null)+1))])
    reference=pd.DataFrame([dict(test="null_distribution",**summary,empirical_p_greater=summary["conditional_p_right"])])
    compare("ed9_population_vector_null_summary",actual,reference,["test"],["observed_mean","null_mean","null_sd","null_quantile025","null_quantile975","empirical_p_greater"],output,analyses)




def run(release_root: Path, output_dir: Path) -> dict[str, Any]:
    release_root,output_dir=Path(release_root),Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    analyses=[];gaps=[]
    for n,fn in [(2,ed2),(3,ed3),(5,ed5),(7,ed7),(9,ed9)]:
        source=release_root/f"extended_data_figure_{n}/source"
        try:
            fn(release_root,source,output_dir,analyses) if n==9 else fn(source,output_dir,analyses)
        except Exception as exc:
            gaps.append(dict(artifact=f"extended_data_figure_{n}",error=repr(exc)))
    result=dict(status="INCOMPLETE" if gaps else "PASS" if all(a["status"]=="PASS" for a in analyses) else "FAIL",analyses=analyses,gaps=gaps,
                method="Frozen derived per-unit values; mouse CR1 and original correction families; acquisition not read; saved statistics only parity targets")
    (output_dir/"ED_STATISTICS_REPORT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--release-root",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.release_root,args.output_dir);print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="PASS" else 1)
