"""Recompute main-figure and ED1 publication statistics from compact source data.

This module never reads raw acquisition data or accepted artwork. ``run`` needs
only the included derived inputs. Saved statistics are comparison authorities,
not estimator inputs.

CR1 follows the frozen producers: G/(G-1)*(N-1)/(N-K), two-sided t(G-1).
BH families are kept separate (Figure1 BH4/BH3/BH6, Figure2 BH4, Figure3
separate first-window and long-gap BH4, Figure4e BH2, ED1 BH18).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster


def read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1)
    return out


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def fit(y, design, clusters) -> dict:
    y, design, clusters = np.asarray(y, float), np.asarray(design, float), np.asarray(clusters, str)
    keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
    y, design, clusters = y[keep], design[keep], clusters[keep]
    model = sm.OLS(y, design).fit()
    covariance = cov_cluster(model, clusters, use_correction=True)
    g = len(np.unique(clusters))
    return {"beta": model.params, "covariance": covariance,
            "se": np.sqrt(np.maximum(np.diag(covariance), 0)), "df": g-1,
            "n_rows": len(y), "n_animals": g, "r2": model.rsquared,
            "ordinary": model}


def contrast(model, vector) -> dict:
    vector = np.asarray(vector, float)
    estimate = float(vector @ model["beta"])
    se = float(np.sqrt(max(vector @ model["covariance"] @ vector, 0)))
    t = estimate / se
    critical = stats.t.ppf(.975, model["df"])
    return dict(estimate=estimate, se=se, t=t, df=model["df"],
                p=float(2*stats.t.sf(abs(t), model["df"])),
                ci_low=estimate-critical*se, ci_high=estimate+critical*se,
                n_rows=model["n_rows"], n_animals=model["n_animals"])


def regression(frame, xcol, ycol, animal) -> dict:
    sub = frame[[xcol, ycol, animal]].dropna()
    x, y = sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)
    model = fit(y, np.column_stack([np.ones(len(x)), x]), sub[animal])
    c = contrast(model, [0, 1])
    p = stats.pearsonr(x, y)
    return dict(intercept=float(model["beta"][0]), slope=c["estimate"],
                slope_se=c["se"], slope_t=c["t"], df=c["df"],
                p_two_sided=c["p"], p_less=float(stats.t.cdf(c["t"],c["df"])),
                p_greater=float(stats.t.sf(c["t"],c["df"])), r2=float(model["r2"]),
                pearson_r=float(p.statistic),
                n_rows=c["n_rows"], n_animals=c["n_animals"],
                ci95_low=c["ci_low"], ci95_high=c["ci_high"])


def mean_ci(values) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    se = float(stats.sem(x)) if len(x)>1 else np.nan
    half = stats.t.ppf(.975, len(x)-1)*se if len(x)>1 else np.nan
    return dict(mean=float(x.mean()), sem=se, lo=float(x.mean()-half),
                hi=float(x.mean()+half), n=len(x))


def signflip(values) -> float:
    x = np.asarray(values, float)
    signs = 2*((np.arange(2**len(x))[:, None] >> np.arange(len(x))) & 1)-1
    return float(np.mean(np.abs(signs @ x / len(x)) >= abs(x.mean())-1e-12))


class Audit:
    def __init__(self, output: Path):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.analyses: list[dict] = []
        self.gaps: list[dict] = []

    def compare(self, name: str, regenerated: pd.DataFrame, saved: pd.DataFrame,
                keys: list[str], columns: list[str], *, atol=1e-10, rtol=1e-9,
                unit: str, family: str, note=""):
        if not columns:
            raise AssertionError(f"{name}: empty numeric comparison contract")
        for side, table in [("regenerated", regenerated), ("reference", saved)]:
            missing = set(keys).union(columns).difference(table.columns)
            if missing or table.empty:
                raise AssertionError(f"{name}: {side} is empty or missing required columns: {sorted(missing)}")
        a = regenerated.set_index(keys).sort_index()
        b = saved.set_index(keys).sort_index()
        checks = []
        if not a.index.equals(b.index):
            raise AssertionError(f"{name}: row identities differ")
        for col in columns:
            av, bv = a[col].to_numpy(float), b[col].to_numpy(float)
            # Tiny p-values are tested relatively, so the absolute tolerance
            # cannot accidentally turn a several-fold probability drift into PASS.
            prob = bool({"p", "q"}.intersection(col.lower().split("_")))
            limit = (1e-300 if prob else atol) + rtol*np.abs(bv)
            same_nan = np.isnan(av) & np.isnan(bv)
            ok = same_nan | (np.isfinite(av) & np.isfinite(bv) & (np.abs(av-bv)<=limit))
            for key, actual, expected, passed in zip(a.index, av, bv, ok):
                checks.append(dict(key=str(key), column=col, regenerated=float(actual),
                                   saved=float(expected), absolute_delta=float(abs(actual-expected)),
                                   pass_check=bool(passed)))
        regenerated.to_csv(self.output/(name+".csv"), index=False)
        comparison = pd.DataFrame(checks)
        comparison.to_csv(self.output/(name+"_verification.csv"), index=False)
        result = dict(analysis=name, status="PASS" if comparison.pass_check.all() else "FAIL",
                      rows=len(regenerated), checks=len(checks), unit=unit, correction_family=family,
                      tolerance=dict(absolute=atol, relative=rtol, tiny_probabilities="relative only"),
                      max_abs_delta=float(comparison.absolute_delta.max()), note=note,
                      regenerated_table=name+".csv", verification_table=name+"_verification.csv")
        self.analyses.append(result)
        return result


def main1(root: Path, audit: Audit):
    p = root/"main_figure_1/source/frozen_source/plot_ready"
    for prefix, group, effect, fold, family in [
        ("reward_acquisition_day1_month", "month", "coef_log_later_minus_naive", "fold_change_later_over_naive", "four naive-versus-return-month contrasts BH4"),
        ("long_bout_group", "group_index", "coef_log_group_b_minus_group_a", "fold_change_group_b_over_group_a", "three pairwise reward-context contrasts BH3")]:
        points = read(p/(prefix+"_points_current_plus_new12.csv"))
        saved = read(p/(prefix+"_clustered_contrasts_current_plus_new12.csv"))
        rows=[]
        for row in saved.itertuples():
            sub=points.loc[points[group].isin([row.group_a,row.group_b])].copy()
            sub=sub.loc[np.isfinite(sub.lb0)&sub.lb0.gt(0)]
            indicator=sub[group].eq(row.group_b).to_numpy(float)
            model=fit(np.log(sub.lb0),np.column_stack([np.ones(len(sub)),indicator]),sub.animal_id)
            c=contrast(model,[0,1])
            rows.append(dict(comparison=row.comparison, **{effect:c["estimate"], fold:np.exp(c["estimate"])},
                             n_sessions=c["n_rows"], n_animals=c["n_animals"], cluster_ols_t=c["t"],
                             cluster_ols_p=c["p"]))
        result=pd.DataFrame(rows)
        result["cluster_ols_q_bh"]=bh(result.cluster_ols_p)
        audit.compare("main1_"+prefix,result,saved,["comparison"],
                      [effect,fold,"n_sessions","n_animals","cluster_ols_t","cluster_ols_p","cluster_ols_q_bh"],
                      unit="session observations; OLS with standard errors clustered by animal; log(lb0) response",family=family,
                      note="Primary panel inference; historical GEE/bootstrap sensitivity columns are not used by the artwork.")
    returns=read(p/"reward_acquisition_day1_month_points_current_plus_new12.csv")
    returns=returns.loc[returns.month.isin([3,6,9,12]) & returns.lb0.gt(0)].copy()
    c=contrast(fit(np.log(returns.lb0), np.column_stack([np.ones(len(returns)), returns.month/3]), returns.animal_id), [0,1])
    trend=pd.DataFrame([dict(analysis="within_return_experience_trend", n_rows=c["n_rows"],
        n_animals=c["n_animals"], estimate_log=c["estimate"], cluster_se=c["se"], t=c["t"], df=c["df"],
        p_two_sided_cluster_t=c["p"], fold_change_per_unit=np.exp(c["estimate"]),
        ci95_log_low=c["ci_low"], ci95_log_high=c["ci_high"])])
    audit.compare("main1_within_return_experience_trend", trend, read(p/"return_experience_statistics.csv"),
        ["analysis"], list(trend.columns[1:]), unit="56 return sessions from 17 animals; log-time slope per return cycle; standard errors clustered by animal",
        family="one captioned trend test; nominal two-sided p")
    points=read(p/"ed04_session_route_current_plus_new12.csv")
    saved=read(p/"ed04_route_stats_current_plus_new12.csv")
    group_saved=read(p/"ed04_route_group_stats_current_plus_new12.csv")
    groups=group_saved.condition.tolist()
    x=np.column_stack([points.condition.eq(g).to_numpy(float) for g in groups])
    model=fit(points.route_fraction_correct,x,points.animal)
    rows=[]
    for row in saved.itertuples():
        cvector=np.zeros(len(groups));cvector[groups.index(row.target_condition)]=1;cvector[groups.index(row.baseline_condition)]=-1
        c=contrast(model,cvector)
        rows.append(dict(comparison=row.comparison,mean_diff_target_minus_baseline=c["estimate"],
                         cluster_se=c["se"],cluster_ci95_low=c["ci_low"],cluster_ci95_high=c["ci_high"],
                         t_cluster=c["t"],df_cluster=c["df"],p_cluster=c["p"],n_rows_total=c["n_rows"],n_clusters_total=c["n_animals"]))
    result=pd.DataFrame(rows);result["q_bh6"]=bh(result.p_cluster)
    audit.compare("main1_route_contrasts",result,saved,["comparison"],list(result.columns[1:]),
                  unit="162 sessions; four-level saturated condition model; standard errors clustered by animal (33 animals)",family="six route-score pairwise contrasts BH6")
    rows=[]
    for i,g in enumerate(groups):
        vector=np.zeros(len(groups));vector[i]=1;c=contrast(model,vector)
        sub=points.loc[points.condition.eq(g)]
        rows.append(dict(condition=g,mean_route_fraction_correct=c["estimate"],cluster_se=c["se"],
                         cluster_ci95_low=c["ci_low"],cluster_ci95_high=c["ci_high"],n_sessions=len(sub),
                         n_animals=sub.animal.nunique(),n_clusters_total=c["n_animals"],df_cluster=c["df"]))
    result=pd.DataFrame(rows)
    audit.compare("main1_route_group_ci",result,group_saved,["condition"],list(result.columns[1:]),
                  unit="condition means from same saturated CR1 model",family="none; 95% marginal t(G-1) intervals")
    for prefix,key in [("reward_acquisition_day1_month","month"),("long_bout_group","group_index")]:
        points=read(p/(prefix+"_points_current_plus_new12.csv"));saved=read(p/(prefix+"_summary_current_plus_new12.csv"))
        rows=[]
        for g,sub in points.groupby(key):
            x=pd.to_numeric(sub.lb0,errors="coerce").dropna().to_numpy(float)
            rows.append({key:g,"mean":x.mean(),"sem":stats.sem(x),"std":x.std(ddof=1),"median":np.median(x),
                         "min":x.min(),"max":x.max(),"n_sessions":len(x),"n_animals":sub.animal_id.nunique()})
        result=pd.DataFrame(rows)
        audit.compare("main1_"+prefix+"_summary",result,saved,[key],list(result.columns[1:]),
                      unit="session-level descriptive mean/SEM",family="none")


def main2(root: Path,audit: Audit):
    p=root/"main_figure_2/source/source_data"
    points=read(p/"consecutive_gap/consecutive_gap_clean_long_gap_corrected_points.csv")
    saved=read(p/"consecutive_gap/consecutive_gap_clean_long_gap_corrected_stats.csv")
    rows=[]
    for metric,sub in points.loc[points.metric.isin(["place_map_correlation", "pv_correlation", "level_map_correlation"])].groupby("metric",sort=False):
        row=regression(sub,"prev_gap_correlation","next_gap_correlation","animal_id")
        # The frozen dense producer used complementary CDF tails, not sf.
        # This matters for the historical ~1e-8 rate p at machine epsilon.
        row["p_greater"]=1-row["p_less"]
        row["p_two_sided"]=2*min(row["p_less"],row["p_greater"])
        rows.append(dict(metric=metric,**row))
    dense=pd.DataFrame(rows)
    points=read(p/"rate_chain/session_pair_rate_consecutive_gap_points_canonical.csv")
    rate=regression(points,"value_sorted_ab","value_sorted_bc","animal_id")
    rate_saved=read(p/"rate_chain/session_pair_rate_consecutive_gap_stats_canonical.csv")
    rate_compare=pd.DataFrame([dict(scope=rate_saved.scope.iloc[0],intercept=rate["intercept"],slope=rate["slope"],
                                  slope_se_cluster_animal=rate["slope_se"],slope_t_cluster_animal=rate["slope_t"],df=rate["df"],
                                  p_two_sided_cluster_animal=rate["p_two_sided"],pearson_r=rate["pearson_r"],n_rows=rate["n_rows"],n_animals=rate["n_animals"])])
    audit.compare("main2_rate_previous_next",rate_compare,rate_saved,["scope"],list(rate_compare.columns[1:]),
                  unit="22 session chains, 11 mice; CR1",family="current Figure2 displayed BH4 recomputed below")
    displayed=dense.loc[dense.metric.isin(["place_map_correlation","pv_correlation","level_map_correlation"])].copy()
    displayed=pd.concat([displayed,pd.DataFrame([dict(metric="ensemble_rate_correlation",**rate)])],ignore_index=True)
    displayed["q_bh4_two_sided"]=bh(displayed.p_two_sided)
    authority=read(root/"main_figure_2/source/statistical_inputs/displayed_estimand_bh4.csv")
    audit.compare("main2_displayed_previous_next_bh4",displayed,authority,["metric"],
                  ["intercept","slope","slope_se","slope_t","df","p_two_sided","n_rows","n_animals","q_bh4_two_sided"],
                  unit="metric-specific neuron/bin or session-chain observations, with standard errors clustered by animal",family="three dense metrics plus canonical pair-level ensemble rate BH4")
    saved=read(p/"distribution/metric_summary.csv");pair=read(p/"distribution/pair_metric_summary.csv")
    rows=[]
    for metric,stem in [("place","figure2_panel_d_place"),("pv","figure2_panel_f_pv"),("rate","figure2_panel_h_rate")]:
        z=np.load(next((p/"distribution").glob(stem+"_shuffle_mean*.npz")))
        null=z["null"].astype(float);obs=float(z["obs"].item())
        sub=pair.loc[pair.metric.eq(metric)]
        rows.append(dict(metric=metric,n_pairs=len(sub),observed_pair_mean=sub.observed_pair_value.mean(),
                         null_mean=null.mean(),null_sd=null.std(ddof=1),
                         p_right_plus_one=(1+np.count_nonzero(null>=obs))/(len(null)+1),common_repeats=len(null)))
    result=pd.DataFrame(rows)
    audit.compare("main2_shuffle_mean_tests",result,saved,["metric"],list(result.columns[1:]),atol=5e-8,rtol=1e-7,
                  unit="equal pair means; supplied 10,000 technical null draws; one-sided plus-one Monte Carlo",family="none",
                  note="Place/PV null arrays are frozen float32, so their descriptive moments use 5e-8 absolute tolerance. Monte Carlo p-values are exact.")
    s=root/"main_figure_2/source/statistical_inputs"
    same=read(s/"same_stage_correlations.csv");same=same.loc[~truth(same.descriptive_only)&same.experience_index.isin([0,1,2,3])]
    sh=read(s/"same_stage_shuffle.csv");sh=sh.loc[~truth(sh.descriptive_only)&sh.experience_index.isin([0,1,2,3])]
    long=read(s/"long_gap_pair_rows.csv")
    curve=read(p/"gap_curve/gap_with_day1_day2_curve_data.csv")
    rows=[]
    mapping={"place":"same_stage_place_map_corr","pv":"same_stage_pv_corr","level":"same_stage_level_corr"}
    for row in curve.itertuples():
        metric=row.metric
        if metric=="rate":
            path=p/"gap_curve"/("adjacent_day_ensemble_rate_pair_values.csv" if row.time_days==1 else "canonical_ensemble_rate_all_gap_pair_values.csv")
            sub=read(path)
            if row.time_days!=1:sub=sub.loc[sub.gap_months.eq(row.gap_months)]
            values=sub.rate_correlation;shuffle=sub.shuffle_mean.mean();animals=sub.animal.nunique()
        elif row.time_days==1:
            values=same[mapping[metric]];shuffle=sh[mapping[metric]+"_shuffle_mean"].mean();animals=same.animal_id.nunique()
        else:
            sub=long.loc[long.metric.eq(metric)&long.gap_months.eq(row.gap_months)]
            values=sub.observed_mean;shuffle=sub.shuffle_mean.mean();animals=sub.animal.nunique()
        ci=mean_ci(values)
        rows.append(dict(metric=metric,time_days=row.time_days,observed_mean=ci["mean"],observed_ci95_low=ci["lo"],observed_ci95_high=ci["hi"],
                         n_animals=animals,n_pairs=ci["n"],shuffle_mean=shuffle))
    result=pd.DataFrame(rows)
    audit.compare("main2_gap_curve_pair_mean_ci",result,curve,["metric","time_days"],list(result.columns[2:]),
                  unit="equal session-pair means; t(N_pairs-1) descriptive 95% intervals",family="none; no gap-wise hypothesis tests used by panel")


def main3(root: Path,audit: Audit):
    p=root/"main_figure_3/source/frozen_source/source_data"
    for folder,pointname,statname,xcol,correction in [
        ("first30","combined_first30_points_all_metrics.csv","combined_first30_stats_all_metrics.csv","experience_index",True),
        ("long_gap","main3_experience_trend_points_no_manual_mask.csv","main3_experience_trend_clustered_stats_no_manual_mask.csv","experience",False)]:
        metrics=["place_map_correlation","level_map_correlation","pv_correlation","ensemble_rate_correlation"]
        points=read(p/folder/pointname);saved=read(p/folder/statname)
        points=points.loc[points.metric.isin(metrics)]
        saved=saved.loc[saved.metric.isin(metrics)]
        rows=[dict(metric=metric,**regression(sub,xcol,"value","animal")) for metric,sub in points.groupby("metric",sort=False)]
        result=pd.DataFrame(rows)
        if correction:
            result["q_bh_two_sided"]=bh(result.p_two_sided)
            result["cluster_df"]=result.df
            cols=["intercept","slope","slope_se","slope_t","p_two_sided","cluster_df","pearson_r","n_rows","n_animals","q_bh_two_sided"]
        else:
            chosen=result.metric.isin(["place_map_correlation","level_map_correlation","pv_correlation","ensemble_rate_correlation"])
            result["q_bh_two_sided"]=np.nan;result.loc[chosen,"q_bh_two_sided"]=bh(result.loc[chosen,"p_two_sided"])
            cols=["intercept","slope","slope_se","slope_t","df","p_two_sided","p_less","p_greater","r2","pearson_r","n_rows","n_animals","q_bh_two_sided"]
        audit.compare("main3_"+folder+"_experience_trends",result,saved,["metric"],cols,
                      unit="pair/animal–experience rows; standard errors clustered by animal",family="separate four-representation BH4 family; decoder sensitivities excluded")
    # The descriptive curves are supplied as mean/SEM/support Source Data.
    # No inferential test uses the full per-animal time courses. The observation
    # tables above retain both first-window and long-gap regression inputs.


def main4(root: Path,audit: Audit):
    p=root/"main_figure_4/source"
    ref=p/"reference_record";saved=read(ref/"plot_ready/figure4_day_source_statistics.csv")
    rows=[]
    for analysis,source,value,animal in [
        ("d1_native_decoder_neuron_id_shuffle",ref/"source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_pair_summary.csv","neuron_id_shuffle_minus_observed","animal_id"),
        ("d1_native_reward_control_advantage",ref/"plot_ready/panel_d_native_reference_rows.csv","control_minus_reward","animal")]:
        sub=read(source)
        if "variant" in sub:sub=sub.loc[sub.variant.eq("d1v1")&truth(sub.included_reward_location_control)]
        if value not in sub:
            sub[value]=sub.neuron_id_shuffle_error-sub.observed_error
        c=contrast(fit(sub[value],np.ones((len(sub),1)),sub[animal]),[1])
        rows.append(dict(analysis=analysis,estimate=c["estimate"],cluster_se=c["se"],cluster_t=c["t"],cluster_df=c["df"],
                         cluster_p=c["p"],ci95_low=c["ci_low"],ci95_high=c["ci_high"],n_pairs=c["n_rows"],n_animals=c["n_animals"]))
    result=pd.DataFrame(rows)
    audit.compare("main4_decoder_advantages",result,saved.loc[saved.analysis.isin(result.analysis)], ["analysis"],list(result.columns[1:]),
                  unit="within-pair advantages; intercept-only OLS with standard errors clustered by animal",family="none; prespecified nominal tests")
    sub=read(ref/"plot_ready/panel_f_matched_pair_means.csv")
    x=sub.d1_reward_proxy_error.to_numpy(float);y=sub.first_long_bout_time_s.to_numpy(float)
    assert len(sub)==30 and sub.animal_id.nunique()==14 and sub.neuron_target_n.eq(166).all()
    xz=(x-x.mean())/x.std(ddof=1);yz=(y-y.mean())/y.std(ddof=1)
    c=contrast(fit(yz,np.column_stack([np.ones(len(x)),xz]),sub.animal_id),[0,1])
    result=pd.DataFrame([dict(analysis="d1_matched_reward_behavior_association",standardized_beta=c["estimate"],pearson_r=stats.pearsonr(x,y).statistic,
                             cluster_se=c["se"],cluster_t=c["t"],cluster_df=c["df"],cluster_p=c["p"],ci95_low=c["ci_low"],ci95_high=c["ci_high"],n_pairs=len(x),n_animals=sub.animal_id.nunique())])
    audit.compare("main4_reward_behavior_k166",result,saved.loc[saved.analysis.eq(result.analysis.iloc[0])],["analysis"],list(result.columns[1:]),
                  unit="30 pair means from 14 mice; behavior ~ standardized reward decoding error; CR1",family="none; raw p, unlike ED10a BH3")
    pairs=read(p/"plot_ready/panel_e_pair_profiles.csv")
    animals=pairs.groupby(["group","animal","bin"],as_index=False)[["pv","pv_sh"]].mean()
    target_animals=read(p/"plot_ready/panel_e_animal_profiles.csv")
    audit.compare("main4_route_animal_profiles",animals,target_animals,["group","animal","bin"],["pv","pv_sh"],
                  unit="pairs within animal, then equally weighted animals",family="none")
    rows=[];effects={}
    curves=[]
    for group,sub in animals.groupby("group",sort=False):
        v=sub.pivot(index="animal",columns="bin",values="pv")
        vs=sub.pivot(index="animal",columns="bin",values="pv_sh")
        effects[group]=v[21]-v.loc[:,0:20].mean(axis=1)
        for item,values in [("port",v[21]),("elsewhere",v.loc[:,0:20].mean(axis=1)),("port minus elsewhere",effects[group]),
                            ("shuffle: port minus elsewhere",vs[21]-vs.loc[:,0:20].mean(axis=1))]:
            ci=mean_ci(values)
            rows.append(dict(group=group,item=item,mean=ci["mean"],lo=ci["lo"],hi=ci["hi"],n_mice=ci["n"],n_positive=int((values>0).sum()),
                             p_exact=signflip(values) if item=="port minus elsewhere" else np.nan))
        for b,bs in sub.groupby("bin"):
            ci=mean_ci(bs.pv)
            curves.append(dict(group=group,bin=b,mean=ci["mean"],lo=ci["lo"],hi=ci["hi"],shuffle=bs.pv_sh.mean(),n_mice=ci["n"]))
    result=pd.DataFrame(rows);result["q_BH"]=np.nan
    mask=result.group.isin(["home_to_reward","reward_to_home"])&result.item.eq("port minus elsewhere")
    result.loc[mask,"q_BH"]=bh(result.loc[mask,"p_exact"])
    saved=read(p/"plot_ready/panel_e_statistics.csv")
    saved=saved.loc[saved.group.isin(["home_to_reward","reward_to_home"])]
    audit.compare("main4_route_enrichment",result,saved,["group","item"],list(result.columns[2:]),
                  unit="animal-mean effects; exact two-sided sign-flip of mean (all 2^G signs); 95% t intervals",family="two shown direction enrichment tests BH2")
    result=pd.DataFrame(curves)
    audit.compare("main4_route_curve",result,read(p/"plot_ready/panel_e_curve.csv"),["group","bin"],list(result.columns[2:]),
                  unit="equal animals; t(G-1) descriptive 95% intervals",family="none")


def main5(root: Path,audit: Audit):
    p=root/"main_figure_5/source/source_data/figure5_bundle"
    points=read(p/"level_map_vs_exploration_efficiency_primary_clean_rows.csv")
    saved=json.loads((p/"level_map_vs_exploration_efficiency_primary_clean_stats.json").read_text())
    # Preserve the estimator direction from the saved producer: neural level
    # correlation as response, exploration efficiency as predictor.
    row=regression(points,"exploration_efficiency","level_map_correlation","animal_id")
    result=pd.DataFrame([dict(analysis="level_exploration",cluster_ols_slope=row["slope"],cluster_ols_se=row["slope_se"],cluster_ols_t=row["slope_t"],
                             cluster_ols_df=row["df"],cluster_ols_p=row["p_two_sided"],ols_intercept=row["intercept"],pearson_r=row["pearson_r"],
                             n_pairs=row["n_rows"],n_animals=row["n_animals"])])
    audit.compare("main5_level_exploration",result,pd.DataFrame([dict(analysis="level_exploration",**saved)]),["analysis"],list(result.columns[1:]),
                  unit="36 session-pair observations, standard errors clustered by animal (14 animals)",family="none; prespecified nominal association")
    p=root/"main_figure_5/source/source_data/level_pv"
    pairs=read(p/"level_pv_pair_matrices.csv")
    matrix=pairs.pivot_table(index="train_depth",columns="test_depth",values="pearson_r",aggfunc="mean").reset_index()
    matrix.columns=[str(c) for c in matrix.columns]
    saved=read(p/"level_pv_matrix.csv");saved.columns=[str(c) for c in saved.columns]
    audit.compare("main5_level_pv_matrix",matrix,saved,["train_depth"],list(matrix.columns[1:]),
                  unit="cell-wise equal session-pair means, 32 or 36 supported pairs",family="none; descriptive heatmap")


def ed1(root: Path,audit: Audit):
    p=root/"extended_data_figure_1/source/plot_ready"
    points=read(p/"ed01_endpoint_axis_session_rows_current153.csv")
    saved=read(p/"ed01_endpoint_axis_cluster_stats_current153.csv")
    saved=saved.loc[saved.axis_key.isin(["maze_time_s","maze_tiles"])].copy()
    rows=[]
    for row in saved.itertuples():
        sub=points.loc[points.endpoint_key.eq(row.endpoint_key)&points.group_key.isin([row.group_a,row.group_b])].copy()
        v=pd.to_numeric(sub[row.axis_key],errors="coerce");sub=sub.loc[np.isfinite(v)&v.ge(0)].copy()
        y=np.log1p(sub[row.axis_key].to_numpy(float));b=sub.group_key.eq(row.group_b).to_numpy(float)
        c=contrast(fit(y,np.column_stack([np.ones(len(y)),b]),sub.animal_id),[0,1])
        avalues=sub.loc[sub.group_key.eq(row.group_a),row.axis_key];bvalues=sub.loc[sub.group_key.eq(row.group_b),row.axis_key]
        rows.append(dict(endpoint_key=row.endpoint_key,axis_key=row.axis_key,group_a=row.group_a,group_b=row.group_b,
                         n_sessions_group_a=len(avalues),n_sessions_group_b=len(bvalues),n_sessions=c["n_rows"],n_mice=c["n_animals"],
                         mean_group_a=avalues.mean(),mean_group_b=bvalues.mean(),raw_mean_diff_group_b_minus_a=bvalues.mean()-avalues.mean(),
                         coef_log1p_group_b_minus_a=c["estimate"],effect_ratio_plus_one=np.exp(c["estimate"]),cluster_t=c["t"],cluster_p_two_sided=c["p"],cluster_df=c["df"]))
    result=pd.DataFrame(rows)
    if len(result)!=18:
        raise AssertionError("ED1 acquisition family must contain 18 displayed time/tile contrasts")
    result["p_bh18"]=bh(result.cluster_p_two_sided)
    audit.compare("ed1_endpoint_bh18",result,saved,["endpoint_key","axis_key","group_a","group_b"],list(result.columns[4:]),
                  unit="eligible session rows, log1p outcomes, standard errors clustered by animal",family="18 displayed endpoint×time/tile-axis×group contrasts BH18")
    points=read(p/"ed02_tile_time_loglog_rows_current153.csv");points=points.loc[truth(points.included_in_loglog_model)]
    row=regression(points,"log_elapsed_time_s","log_tile_transitions","animal_id")
    result=pd.DataFrame([dict(analysis="tile_time",intercept_log_scale=row["intercept"],slope_loglog=row["slope"],cluster_se=row["slope_se"],t_cluster=row["slope_t"],
                             p_cluster_t_two_sided=row["p_two_sided"],ci95_slope_low=row["ci95_low"],ci95_slope_high=row["ci95_high"],
                             n_sessions=row["n_rows"],n_mice=row["n_animals"],df_t=row["df"])] )
    saved=read(p/"ed02_tile_time_loglog_stats_current153.csv");saved["analysis"]="tile_time"
    audit.compare("ed1_tile_time",result,saved,["analysis"],list(result.columns[1:]),
                  unit="153 sessions; log-log OLS, standard errors clustered by animal",family="none; single prespecified diagnostic")
    points=read(p/"ed01_panel_h_first30_session_rows_cap60.csv")
    points=points.loc[points.window_id.eq("first_30_min")&points.analysis_mode.isin(["explore","reward"])&points.day.isin(["day1","day2"])]
    rows=[]
    for keys,sub in points.groupby(["analysis_mode","day","month"]):
        sub=sub.loc[np.isfinite(pd.to_numeric(sub.mode_time_distance_rate_cm_s,errors="coerce"))]
        ci=mean_ci(sub.mode_time_distance_rate_cm_s)
        rows.append(dict(zip(["analysis_mode","day","month"],keys),speed_mean_cm_s=ci["mean"],speed_sem_cm_s=ci["sem"],n_sessions=len(sub),n_animals=sub.animal_id.nunique()))
    result=pd.DataFrame(rows);saved=read(p/"ed01_panel_h_first30_group_summary_cap60.csv")
    saved=saved.loc[saved.summary_variant.eq("exact_current_mode_rule")&saved.window_id.eq("first_30_min")&saved.analysis_mode.isin(["explore","reward"])&saved.day.isin(["day1","day2"])]
    audit.compare("ed1_speed_mean_sem",result,saved,["analysis_mode","day","month"],list(result.columns[3:]),
                  unit="session speeds, first 30 min in-maze, 60 cm/s segment cap; mean/SEM",family="none; descriptive only")




def run(release_root: Path,output_dir: Path) -> dict[str,Any]:
    release_root,output_dir=Path(release_root).resolve(),Path(output_dir).resolve()
    audit=Audit(output_dir)
    errors=[]
    for name,function in [("main1",main1),("main2",main2),("main3",main3),("main4",main4),("main5",main5),("ed1",ed1)]:
        try:function(release_root,audit)
        except Exception as exc:
            errors.append(dict(artifact=name,error=f"{type(exc).__name__}: {exc}"))
    status="FAIL" if errors or any(x["status"]=="FAIL" for x in audit.analyses) else "INCOMPLETE" if audit.gaps else "PASS"
    report=dict(schema="source-release-main-statistics/v1",status=status,analyses=audit.analyses,gaps=audit.gaps,errors=errors,
                accepted_artwork_read=False,raw_acquisition_read=False,
                scope="publication hypothesis tests and identified descriptive moments, not every historical sensitivity table")
    (output_dir/"REPORT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.release_root,args.output_dir)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__=="__main__":main()
