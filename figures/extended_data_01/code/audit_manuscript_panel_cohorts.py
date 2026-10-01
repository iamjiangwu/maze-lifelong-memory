"""Build a panel-by-panel manuscript cohort and exclusion audit.

This script is intentionally tied to the pointer-selected 2026-08-15 figure
records.  It does not modify figures or captions.  It freezes the exact unit
lists used by the displayed panels, expands shared cohort definitions to all
128 captioned panels, and records discrepancies that require adjudication.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "manuscript_panel_cohort_audit_2026-08-15"

F1 = ROOT / "docs/figure_version_records/main_figure1_2026-08-13_round12_no_port_mouse1636"
F2 = ROOT / "docs/figure_version_records/main_figure2_2026-08-13_round5_pair_mean_day1day2"
F3 = ROOT / "docs/figure_version_records/main_figure3_2026-08-16_round9_available_window_slopes"
F4 = ROOT / "docs/figure_version_records/main_figure4_2026-08-15_round7_neuron_id_shuffle"
F5 = ROOT / "docs/figure_version_records/main_figure5_2026-08-15_round8_neuron_id_shuffle"
E1 = ROOT / "docs/figure_version_records/extended_data_fig01_2026-08-17_round19_canonical_animal_clusters"
E2 = ROOT / "docs/figure_version_records_archive_2026-08-09/extended_data_fig02_2026-08-12_round23_preceding_turn"
E3 = ROOT / "docs/figure_version_records/extended_data_fig03_2026-08-13_round3_latent_day_labels"
E4 = ROOT / "docs/figure_version_records/extended_data_fig04_2026-08-12_round4_cycle_labels"
E5 = ROOT / "docs/figure_version_records/extended_data_fig05_2026-08-16_round16_bh3"
E6 = ROOT / "docs/figure_version_records/extended_data_fig06_2026-08-16_round7_available_curve_all59"
E7 = ROOT / "docs/figure_version_records/extended_data_fig07_2026-08-16_round16_upright_i_axis"
E8 = ROOT / "docs/figure_version_records/extended_data_fig08_2026-08-17_round20_route_field_threshold_axis"
E9 = ROOT / "docs/figure_version_records_archive_2026-08-09/extended_data_fig09_2026-08-14_round32_bh4_correlations"
E10 = ROOT / "docs/figure_version_records/extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def normalize_animal(value: str) -> str:
    match = re.match(r"\d+", str(value).strip())
    if not match:
        raise ValueError(f"Cannot normalize animal ID: {value!r}")
    return str(int(match.group(0)))


def animal_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    animal_id: str


@dataclass
class Cohort:
    cohort_id: str
    label: str
    units: list[Unit]
    source: str
    restriction: str = ""

    @property
    def animals(self) -> list[str]:
        return sorted({u.animal_id for u in self.units}, key=animal_sort_key)


def cohort_from_rows(
    cohort_id: str,
    label: str,
    rows: Iterable[dict[str, str]],
    source: Path,
    unit_fn: Callable[[dict[str, str]], str],
    animal_fn: Callable[[dict[str, str]], str],
    restriction: str = "",
) -> Cohort:
    dedup: dict[str, Unit] = {}
    for row in rows:
        unit_id = unit_fn(row)
        animal_id = normalize_animal(animal_fn(row))
        if unit_id:
            dedup[unit_id] = Unit(unit_id, animal_id)
    units = sorted(dedup.values(), key=lambda u: (animal_sort_key(u.animal_id), u.unit_id))
    return Cohort(cohort_id, label, units, rel(source), restriction)


def union_cohort(cohort_id: str, label: str, cohorts: Iterable[Cohort], source: str) -> Cohort:
    dedup: dict[str, Unit] = {}
    restrictions: list[str] = []
    for cohort in cohorts:
        for unit in cohort.units:
            dedup[unit.unit_id] = unit
        if cohort.restriction:
            restrictions.append(cohort.restriction)
    return Cohort(
        cohort_id,
        label,
        sorted(dedup.values(), key=lambda u: (animal_sort_key(u.animal_id), u.unit_id)),
        source,
        "; ".join(dict.fromkeys(restrictions)),
    )


def subset_cohort(
    cohort_id: str,
    label: str,
    cohort: Cohort,
    keep: Callable[[Unit], bool],
    source: str,
    restriction: str,
) -> Cohort:
    return Cohort(cohort_id, label, [u for u in cohort.units if keep(u)], source, restriction)


def build_cohorts() -> dict[str, Cohort]:
    cohorts: dict[str, Cohort] = {}

    f1_eff_path = F1 / "plot_ready/exploration_first_chunk_current_plus_new_points.csv"
    f1_eff = read_csv(f1_eff_path)
    cohorts["B_N23"] = cohort_from_rows(
        "B_N23", "23 rewarded naive Day-1 behavior sessions", [r for r in f1_eff if r["group"].startswith("Naive")],
        f1_eff_path, lambda r: r["nickname"], lambda r: r["animal_id"]
    )
    cohorts["B_R56"] = cohort_from_rows(
        "B_R56", "56 fixed-month return Day-1 sessions in the Figure-1 roster", [r for r in f1_eff if r["group"].startswith("LTM")],
        f1_eff_path, lambda r: r["nickname"], lambda r: r["animal_id"],
        "Intentionally restricted to sessions with unambiguous 3/6/9/12-month group labels; the special 1616bday1 and 1616b-3day1 sequence is outside this fixed-month behavior universe."
    )

    route_path = F1 / "plot_ready/ed04_session_route_current_plus_new12.csv"
    route = read_csv(route_path)
    cohorts["B_D74"] = cohort_from_rows(
        "B_D74", "74 second-day behavior sessions", [r for r in route if r["condition"] == "experienced_day2"],
        route_path, lambda r: r["session_id"], lambda r: r["animal"]
    )
    cohorts["B_L9"] = cohort_from_rows(
        "B_L9", "9 latent-reward Day-2 sessions", [r for r in route if r["condition"] == "latent_day2"],
        route_path, lambda r: r["session_id"], lambda r: r["animal"]
    )
    cohorts["B_ALL153"] = union_cohort(
        "B_ALL153", "153-session behavioral registry", [cohorts["B_N23"], cohorts["B_R56"], cohorts["B_D74"]],
        rel(route_path)
    )
    cohorts["B_ROUTE162"] = union_cohort(
        "B_ROUTE162", "162 sessions with six route choices resolved", [cohorts["B_ALL153"], cohorts["B_L9"]],
        rel(route_path)
    )
    cohorts["B_EFF152"] = subset_cohort(
        "B_EFF152", "152 sessions reaching 32 unique endpoints", cohorts["B_ALL153"],
        lambda u: u.unit_id != "646-9day2", rel(E1 / "drawn_values/TIME_EFFICIENCY_SUMMARY_ROUND18.csv"),
        "646-9day2 never reached 32 unique endpoints, so 32/N32 is undefined."
    )

    e1_h_path = E1 / "plot_ready/ed01_panel_h_matched153_session_registry.csv"
    cohorts["B_H153"] = cohort_from_rows(
        "B_H153", "current matched-153 ED1h session registry", read_csv(e1_h_path), e1_h_path,
        lambda r: r["session_id"], lambda r: r["animal_id"],
        "Round 18 replaced the historical 120-session cache with the exact 153-session ED1 registry."
    )

    e2_b_path = E2 / "plot_ready/panel_b_session_points.csv"
    e2_b = read_csv(e2_b_path)
    cohorts["B_PN22"] = cohort_from_rows(
        "B_PN22", "22 naive sessions passing the strict joint policy-plus-depth gate",
        [r for r in e2_b if r["group"] == "naive"], e2_b_path,
        lambda r: r["nickname"], lambda r: r["animal_id"],
        "640day1 is removed only because shallow depth support is 2 (<10), although all four non-depth policies are estimable."
    )
    cohorts["B_R58"] = cohort_from_rows(
        "B_R58", "58 pooled return sessions in the policy authority", [r for r in e2_b if r["group"] == "LTM"],
        e2_b_path, lambda r: r["nickname"], lambda r: r["animal_id"],
        "ED2 pools return sessions and does not require assignment to a fixed 3/6/9/12-month group, so both special animal-1616 sessions are eligible."
    )
    e2_c_path = E2 / "plot_ready/panel_c_session_points.csv"
    e2_c = read_csv(e2_c_path)
    cohorts["B_PC132"] = cohort_from_rows(
        "B_PC132", "58 return plus 74 second-day policy sessions", e2_c, e2_c_path,
        lambda r: r["nickname"], lambda r: r["animal_id"]
    )
    cohorts["B_POLICY155"] = union_cohort(
        "B_POLICY155", "23 naive, 58 return and 74 second-day policy sessions",
        [cohorts["B_N23"], cohorts["B_R58"], cohorts["B_D74"]], rel(E2 / "plot_ready/panel_e_distance_from_return.csv")
    )
    e2_f_path = E2 / "plot_ready/panel_f_session_points.csv"
    cohorts["B_SIM152"] = cohort_from_rows(
        "B_SIM152", "23 naive, 58 return and 71 second-day simulation sessions", read_csv(e2_f_path), e2_f_path,
        lambda r: r["nickname"], lambda r: r["animal_id"],
        "837-12day2, 838-12day2 and 839-12day2 lack the maintained half-coverage scalar."
    )
    e2_g_path = E2 / "plot_ready/panel_g_supported_session_rows.csv"
    cohorts["B_DEPTH154"] = cohort_from_rows(
        "B_DEPTH154", "22 naive, 58 return and 74 second-day depth-supported sessions", read_csv(e2_g_path), e2_g_path,
        lambda r: r["nickname"], lambda r: r["animal_id"],
        "640day1 has only 2 shallow-depth events (<10)."
    )
    cohorts["B_PB80"] = union_cohort(
        "B_PB80", "22 naive plus 58 return strict policy sessions", [cohorts["B_PN22"], cohorts["B_R58"]], rel(e2_b_path)
    )

    fluency_path = E3 / "plot_ready/fluency_plot_rows.csv"
    fluency = [r for r in read_csv(fluency_path) if truth(r["included_for_displayed_primary"])]
    claim_map = {
        "B_FLU_A63": "ed4_broad_pre_reward_explore",
        "B_FLU_B65": "ed4_broad_post_reward_explore",
        "B_FLU_C61": "ed4_pre_reward_direct_chain_specificity_check",
        "B_FLU_D65": "ed4_reward_approach_after_reward",
    }
    for cohort_id, claim_id in claim_map.items():
        rows = [r for r in fluency if r["claim_id"] == claim_id]
        cohorts[cohort_id] = cohort_from_rows(
            cohort_id, f"Displayed ED3 fluency rows for {claim_id}", rows, fluency_path,
            lambda r: r["session_id"], lambda r: r["animal"]
        )

    clean36_path = F2 / "source_data/distribution/current_panel_base_pairs_corrected_36_no_1616_short.csv"
    cohorts["N_C36"] = cohort_from_rows(
        "N_C36", "canonical 36 long-gap session pairs", read_csv(clean36_path), clean36_path,
        lambda r: r["pair_key"], lambda r: r["animal"],
        "Excludes the 14-day 1616day1->1616bday1 pair from primary long-gap analyses."
    )
    reward30_path = E10 / "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
    cohorts["N_R30"] = cohort_from_rows(
        "N_R30", "30 rewarded-source long-gap pairs", read_csv(reward30_path), reward30_path,
        lambda r: r["pair_key"], lambda r: r["animal_id"],
        "Six unrewarded-template source pairs (422, 837, 838, 839, 842 and 862 at 0->3 months) are ineligible for reward-location analyses."
    )
    same58_path = E7 / "plot_ready/ED07_FIRST30_PLOT_READY_MAINTAINED58.csv"
    same58_rows = read_csv(same58_path)
    cohorts["N_SAME58"] = cohort_from_rows(
        "N_SAME58", "58 adjacent-day recording cycles with age metadata", same58_rows, same58_path,
        lambda r: f"{r['session_day1']}->{r['session_day2']}", lambda r: r["animal_id"],
        "Animal 841 is excluded because date of birth is unavailable for the age model."
    )
    same59_units = list(cohorts["N_SAME58"].units) + [Unit("841day1->841day2", "841")]
    cohorts["N_SAME59"] = Cohort(
        "N_SAME59", "59 adjacent-day recording cycles", sorted(same59_units, key=lambda u: (animal_sort_key(u.animal_id), u.unit_id)),
        rel(F3 / "source_data/first30/combined_first30_points_all_metrics.csv")
    )
    pivot_excluded = {"837-12day1->837-12day2", "838day1->838day2", "839day1->839day2"}
    cohorts["N_PIV56"] = subset_cohort(
        "N_PIV56", "56 pairs supporting a 30-min sliding pivot", cohorts["N_SAME59"],
        lambda u: u.unit_id not in pivot_excluded, rel(E6 / "plot_ready/ed07_pivot_experience_group_curve_summary_current59.csv"),
        "837-12, 838 naive and 839 naive cycles have <30 min of early maze-time span."
    )
    successive_path = F3 / "source_data/long_gap/main3_experience_trend_points_no_manual_mask.csv"
    cohorts["N_S37"] = cohort_from_rows(
        "N_S37", "37 successive-cycle pairs", read_csv(successive_path), successive_path,
        lambda r: f"{r['src']}->{r['dst']}", lambda r: r["animal"],
        "Intentionally includes the disclosed 14-day 1616day1->1616bday1 pair because the estimand is cycle-dependent rather than a fixed-month gap comparison."
    )
    chain_path = F2 / "source_data/consecutive_gap/consecutive_gap_clean_long_gap_22_chain_scope.csv"
    cohorts["N_CHAIN22"] = cohort_from_rows(
        "N_CHAIN22", "22 clean consecutive long-gap three-session chains", read_csv(chain_path), chain_path,
        lambda r: f"animal{normalize_animal(r['animal_id'])}_{r['triplet']}", lambda r: r["animal_id"],
        "Requires two consecutive long-gap edges; excludes the short 1616 chain and animals without a second long-gap edge."
    )
    gap47_path = ROOT / "docs/sorted_data_full_figure_rerun_2026-06-11/figure4d_same_stage_support_matched_sorted_target164_no_manual_mask_2026-08-07/panel_same_stage_correlations.csv"
    gap47_rows = [r for r in read_csv(gap47_path) if not truth(r["descriptive_only"])]
    cohorts["N_GAP47"] = cohort_from_rows(
        "N_GAP47", "47 inferential Day1-Day2 cycle pairs in the 14-animal long-gap cohort", gap47_rows, gap47_path,
        lambda r: f"animal{normalize_animal(r['animal_id'])}_experience{r['experience_index']}", lambda r: r["animal_id"],
        "Four 12-month same-stage rows are tagged descriptive-only in the source authority."
    )

    clean32_path = E9 / "source_data/panel_f_inference_round24/be_roles/B2_EXCLUSION_CLEAN32_PAIR_CONTRASTS.csv"
    cohorts["N_C32"] = cohort_from_rows(
        "N_C32", "ED9f 32-pair inference cohort", read_csv(clean32_path), clean32_path,
        lambda r: r["pair_id"].replace("__", "->"), lambda r: r["animal"],
        "Excludes four support-valid 9->12-month pairs labelled descriptive_9to12 without a panel-specific scientific justification."
    )

    e8_b_path = E8 / "source_data/ED08_PANEL_B_ALL_ELIGIBLE_PAIR_VALUES.csv"
    e8_b_rows = read_csv(e8_b_path)
    for variant, cohort_id, label in [
        ("d1v1", "N_E8B_D1_30", "30 Day-i decoder-control pairs"),
        ("d2v1", "N_E8B_D2_36", "36 Day-i+1 decoder-control pairs"),
    ]:
        rows = [
            r
            for r in e8_b_rows
            if r["variant"] == variant
            and r["resample_method"] == "frame_balanced"
            and truth(r["support_ok"])
            and int(r["n_draws_valid"]) > 0
        ]
        cohorts[cohort_id] = cohort_from_rows(
            cohort_id, label, rows, e8_b_path, lambda r: r["pair_id"], lambda r: r["animal"],
            "Reward-context eligibility plus finite valid balanced decoder output; source exposure is metadata, not a gate."
        )
    e8_c_path = E8 / "plot_ready/ed08_panel_c_pair_values.csv"
    e8_c_rows = read_csv(e8_c_path)
    cohorts["N_E8C_D1_30"] = cohort_from_rows(
        "N_E8C_D1_30", "30 D1 reward-location PV pairs", [r for r in e8_c_rows if r["pair_variant"] == "d1v1"],
        e8_c_path, lambda r: r["pair_id"], lambda r: r["animal"], "Unrewarded source templates are ineligible."
    )
    cohorts["N_E8C_D2_36"] = cohort_from_rows(
        "N_E8C_D2_36", "36 D2 reward-location PV pairs", [r for r in e8_c_rows if r["pair_variant"] == "d2v1"],
        e8_c_path, lambda r: r["pair_id"], lambda r: r["animal"]
    )
    route_source_path = E8 / "source_data/ed08_route_pair_rows_neural_exclusion_clean.csv"
    route_rows = read_csv(route_source_path)
    for run_label, cohort_id, label in [
        ("home_to_water", "N_ROUTE_H29", "29 home-to-water route pairs"),
        ("water_to_home", "N_ROUTE_R28", "28 water-to-home route pairs"),
    ]:
        rows = [r for r in route_rows if r["run_label"] == run_label and truth(r["passes_min_runs_each_session"])]
        cohorts[cohort_id] = cohort_from_rows(
            cohort_id, label, rows, route_source_path, lambda r: r["pair_id"], lambda r: r["animal"],
            "At least five eligible route runs in each session; zero-run pairs are absent upstream rather than frozen as explicit zero-support rows."
        )
    route_selected_path = E8 / "plot_ready/ed08_panel_alt_pair_values.csv"
    route_selected_rows = read_csv(route_selected_path)
    for run_label, cohort_id, label in [
        ("2_to_water", "N_ROUTE_2_30", "30 rewarded-source node-2-to-water route pairs"),
        ("3_to_water", "N_ROUTE_3_29", "29 rewarded-source node-3-to-water route pairs"),
    ]:
        rows = [r for r in route_selected_rows if r["run_label"] == run_label]
        cohorts[cohort_id] = cohort_from_rows(
            cohort_id,
            label,
            rows,
            route_selected_path,
            lambda r: r["canonical_pair_id"],
            lambda r: r["animal"],
            "Reward-experienced Day-i source followed by at least five eligible route runs in each session.",
        )
    outcome_path = E8 / "plot_ready/ed08_panel_g_pair_values.csv"
    cohorts["N_OUT25"] = cohort_from_rows(
        "N_OUT25", "25 pairs with both rewarded and timeout outcomes", read_csv(outcome_path), outcome_path,
        lambda r: r["pair_id"], lambda r: r["animal"],
        "Both outcome classes are required; animals 422 and 434 have no eligible pair."
    )

    return cohorts


@dataclass
class Panel:
    figure: str
    panel: str
    kind: str
    statistical_unit: str
    cohort_ids: str
    cohort_breakdown: str
    exclusion_class: str
    exclusion_reason: str
    verdict: str
    action_required: bool
    source: str
    note: str = ""


def build_panels() -> list[Panel]:
    panels: list[Panel] = []

    def add(
        figure: str, panel: str, kind: str, statistical_unit: str, cohort_ids: str,
        breakdown: str, exclusion_class: str, reason: str, verdict: str,
        source: str, action: bool = False, note: str = ""
    ) -> None:
        panels.append(Panel(figure, panel, kind, statistical_unit, cohort_ids, breakdown,
                            exclusion_class, reason, verdict, action, source, note))

    def many(figure: str, letters: str, **kwargs: str | bool) -> None:
        for letter in letters.split():
            add(figure, letter, **kwargs)  # type: ignore[arg-type]

    # Main Figure 1
    add("Figure 1", "a", "schematic", "none", "", "none", "not applicable", "diagram", "NOT_APPLICABLE", rel(F1 / "plot_ready/figure1_panel_a_decision_points.csv"))
    add("Figure 1", "b", "schematic", "none", "", "none", "not applicable", "longitudinal design diagram", "NOT_APPLICABLE", rel(F1 / "source_code/preview_figure1_assembly.py"))
    add("Figure 1", "c", "illustrative example", "session", "", "4 sessions; mouse 434", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(F1 / "plot_ready/figure1_endnode_timelines.csv"), note="434RRday1, 434RRday2, 434-3day1, 434-6day1")
    add("Figure 1", "d", "quantitative", "session", "B_N23;B_R56", "23 naive; 17/17/12/10 return by month", "valid fixed-month design restriction", "R56 contains only sessions assignable to the predefined 3/6/9/12-month groups; the special animal-1616 sequence is intentionally outside that universe", "VALID", rel(F1 / "plot_ready/reward_acquisition_day1_month_points_current_plus_new12.csv"))
    add("Figure 1", "e", "quantitative", "session", "B_N23;B_R56", "23 naive + 56 fixed-month returns", "valid fixed-month design restriction", "the pooled display uses the union of the predefined fixed-month return groups, not every session that can be called a return", "VALID", rel(F1 / "plot_ready/exploration_first_chunk_current_plus_new_points.csv"))
    add("Figure 1", "f", "illustrative example", "session", "", "2 sessions; mouse 1636", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(F1 / "plot_ready/figure1_endnode_timelines.csv"), note="1636Rday1, 1636Rday2")
    add("Figure 1", "g", "quantitative", "session", "B_N23;B_L9;B_R56", "23 naive + 9 latent Day2 + 56 fixed-month returns", "valid fixed-month design restriction", "return sessions are drawn from the predefined 3/6/9/12-month groups", "VALID", rel(F1 / "plot_ready/long_bout_group_points_current_plus_new12.csv"))
    add("Figure 1", "h", "quantitative", "session", "B_ROUTE162", "23 naive + 74 second-day + 56 fixed-month returns + 9 latent", "valid fixed-month design restriction", "all included rows resolve all six choices; the return stratum is the predefined fixed-month cohort", "VALID", rel(F1 / "plot_ready/ed04_session_route_current_plus_new12.csv"))

    # Main Figure 2
    add("Figure 2", "a", "illustrative example", "field of view", "", "mouse 422; two sessions", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(F2 / "panel_a_sources/422_corr.fig"), note="422RRday1 and 422-3day1")
    add("Figure 2", "b", "illustrative examples", "matched neuron", "", "4 neurons from 3 pair instances", "author-selected examples", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(F2 / "panel_b_sources"), note="418-3->6; 451-6->9; two neurons from 837-6->9")
    many("Figure 2", "c h m", kind="schematic", statistical_unit="none", cohort_ids="", breakdown="none", exclusion_class="not applicable", reason="metric definition", verdict="NOT_APPLICABLE", source=rel(F2 / "source_code/render_main_figs_v3_publication.py"))
    many("Figure 2", "d e i j n o", kind="quantitative", statistical_unit="long-gap session pair", cohort_ids="N_C36", breakdown="36 pairs / 14 animals", exclusion_class="valid design restriction", reason="canonical clean36 excludes only the 14-day 1616 pair", verdict="VALID", source=rel(F2 / "source_data/distribution/current_panel_base_pairs_corrected_36_no_1616_short.csv"))
    many("Figure 2", "f k p", kind="quantitative curve", statistical_unit="session pair", cohort_ids="N_GAP47;N_C36", breakdown="47 pairs at 1 day; 36/22/11/4 at 3/6/9/12 months", exclusion_class="design plus longitudinal availability", reason="1-day point is restricted to the 14-animal long-gap cohort; later losses reflect follow-up availability", verdict="VALID_WITH_DOCUMENTED_SCOPE", source=rel(F2 / "source_data/gap_curve/gap_with_day1_day2_curve_data.csv"))
    many("Figure 2", "g l q", kind="quantitative", statistical_unit="three-session chain", cohort_ids="N_CHAIN22", breakdown="22 chains / 11 animals", exclusion_class="valid design/support restriction", reason="requires two consecutive clean long-gap edges with metric-native finite support", verdict="VALID", source=rel(F2 / "source_data/consecutive_gap/consecutive_gap_clean_long_gap_22_chain_scope.csv"))

    # Main Figure 3
    add("Figure 3", "a", "illustrative example", "matched neuron", "", "mouse 451, 3-to-6-month pair", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(F3 / "source_data/example/451-3day1_to_451-6day1__row0161_A190_B267.npz"))
    many("Figure 3", "b c d", kind="schematic", statistical_unit="none", cohort_ids="", breakdown="none", exclusion_class="not applicable", reason="analysis diagram", verdict="NOT_APPLICABLE", source=rel(F3 / "source_code/render_main_figs_v3_publication.py"))
    many("Figure 3", "e f h i k l", kind="quantitative", statistical_unit="animal-cycle pair", cohort_ids="N_SAME59", breakdown="59 cycles / 22 animals", exclusion_class="none", reason="adjacent-day joint-deconvolution identity; no cross-session CIMap or age metadata needed", verdict="VALID_CONTRACT_CLARIFICATION", source=rel(F3 / "source_data/first30/combined_first30_points_all_metrics.csv"), note="Animal 841 is valid for this estimand but is labelled behavior-only in unrelated long-gap contracts; clarify scope terminology.")
    many("Figure 3", "g j m", kind="quantitative", statistical_unit="successive-cycle session pair", cohort_ids="N_S37", breakdown="37 pairs / 14 animals", exclusion_class="valid cycle-dependent scope", reason="includes one disclosed 14-day pair because the x-axis is recording-cycle number rather than a fixed 3/6/9/12-month gap", verdict="VALID", source=rel(F3 / "source_data/long_gap/main3_experience_trend_points_no_manual_mask.csv"))

    # Main Figure 4
    add("Figure 4", "a", "schematic", "none", "", "none", "not applicable", "decoder diagram", "NOT_APPLICABLE", rel(F4 / "individual_panels/figure4_panel_a_d1.svg"))
    add("Figure 4", "b", "quantitative", "long-gap session pair", "N_C36", "36 pairs / 14 animals", "valid design restriction", "canonical clean36", "VALID", rel(F4 / "source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_pair_summary.csv"))
    add("Figure 4", "c", "quantitative example", "spatial bin", "", "579RLday1->579-3day1", "limited bin support", "six bins with <5 target frames are explicitly blank", "VALID_EXAMPLE", rel(F4 / "plot_ready/panel_c_example_map_rows.csv"))
    add("Figure 4", "d", "quantitative", "long-gap session pair", "N_R30", "30 pairs / 14 animals", "valid reward-context restriction", "source Day i must have rewarded experience", "VALID", rel(F4 / "plot_ready/panel_d_native_reference_rows.csv"))
    add("Figure 4", "e", "quantitative", "matched neuron nested in pair", "N_ROUTE_H29;N_ROUTE_R28", "caption: 29 approach pairs / 13 animals; 28 return pairs / 14 animals", "provenance gap", "active record freezes only an SVG and no direct animal/session source table; exact lists are recoverable only from the ED8 successor route authority", "SOURCE_LEDGER_REQUIRED", rel(F4 / "panel_e_source/figure4_panel_e_source.svg"), True)
    add("Figure 4", "f", "quantitative", "long-gap session pair", "N_R30", "30 pairs / 14 animals", "valid reward-context restriction", "same rewarded-source cohort as panel d", "VALID", rel(F4 / "plot_ready/panel_f_matched_pair_means.csv"))

    # Main Figure 5
    add("Figure 5", "a", "schematic", "none", "", "none", "not applicable", "tree-depth representation diagram", "NOT_APPLICABLE", rel(F5 / "source_code/preview_figure3_assembly.py"))
    many("Figure 5", "b c d e f", kind="quantitative", statistical_unit="long-gap session pair / nested neuron", cohort_ids="N_C36", breakdown="36 pairs / 14 animals; 16,901 matched-neuron observations where applicable", exclusion_class="valid design/activity restriction", reason="canonical clean36; neuron-level panels apply the stated activity gate", verdict="VALID", source=rel(F5 / "source_data/figure5_bundle/displayed_panel_b_heatmap_pair_source_audit.csv"))

    # ED1
    add("ED1", "a", "illustrative example", "session", "", "mouse 434; four sessions", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(E1 / "drawn_values/PANEL_A_TRAJECTORY_PLOT_DATA_ROUND17.csv.gz"))
    many("ED1", "b c d e", kind="quantitative", statistical_unit="session", cohort_ids="B_ALL153", breakdown="23 naive + 56 fixed-month returns + 74 second-day", exclusion_class="valid fixed-month design plus display support", reason="R56 is the predefined 3/6/9/12-month return cohort; the 80% or common-range rule limits the displayed x-range without removing its sessions", verdict="VALID", source=rel(E1 / "plot_ready/ed01_tile_accumulation_whole_group_common_support_current153.csv"))
    add("ED1", "f", "quantitative", "session", "B_ALL153", "153 sessions / 29 biological animals", "none", "Round 19 canonicalizes recording aliases before CR1 clustered inference and retains one 27-test BH family", "VALID", rel(E1 / "plot_ready/ed01_endpoint_axis_session_rows_current153.csv"))
    add("ED1", "g", "quantitative", "session", "B_ALL153", "153 sessions / 29 biological animals", "none", "Round 19 canonicalizes recording aliases before the log-log CR1 regression; df=28", "VALID", rel(E1 / "plot_ready/ed02_tile_time_loglog_rows_current153.csv"))
    add("ED1", "h", "quantitative", "session", "B_H153", "153 sessions matched to panels b-g", "none", "Round 18 replaced the historical cache with the exact ED1 registry", "VALID", rel(E1 / "plot_ready/ed01_panel_h_matched153_speed_by_session_window.csv"))
    add("ED1", "i", "illustrative example", "session", "", "mouse 451 naive and 9-month sessions", "author-selected example", "not an inferential cohort", "ILLUSTRATIVE_ONLY", rel(E1 / "drawn_values/PANEL_I_CYCLE_SUMMARY_ROUND17.csv"))
    many("ED1", "j k", kind="quantitative curve", statistical_unit="session", cohort_ids="B_EFF152", breakdown="152 sessions", exclusion_class="valid fixed-month and estimand-support restrictions", reason="R56 is the fixed-month return cohort; 646-9day2 is additionally excluded because it never reaches 32 unique endpoints", verdict="VALID", source=rel(E1 / "drawn_values/TIME_EFFICIENCY_SUMMARY_ROUND17.csv"))

    # ED2
    add("ED2", "a", "schematic", "none", "", "none", "not applicable", "policy diagram", "NOT_APPLICABLE", rel(E2 / "plot_ready/panel_a_schematic_rows.csv"))
    add("ED2", "b", "quantitative", "session", "B_PB80", "22 naive + 58 return", "caption/criterion mismatch", "640day1 meets the caption's all-four-policies criterion but is removed by an unstated shallow-depth common-cohort gate (2 < 10)", "CAPTION_OR_COHORT_FIX_REQUIRED", rel(E2 / "plot_ready/panel_b_session_points.csv"), True)
    add("ED2", "c", "quantitative", "session", "B_PC132", "58 return + 74 second-day", "none", "all relevant policy rows retained", "VALID", rel(E2 / "plot_ready/panel_c_session_points.csv"))
    add("ED2", "d", "quantitative simulation", "return session / substituted group mean", "B_R58;B_PN22", "58 return; naive substitutions use strict n=22 mean", "inherited cohort mismatch", "naive substitution mean unnecessarily excludes 640 for the four-policy estimand", "RERUN_OR_JUSTIFY_COMMON_COHORT", rel(E2 / "plot_ready/panel_d_policy_contributions.csv"), True)
    add("ED2", "e", "quantitative descriptive", "session", "B_POLICY155", "23 naive + 58 return + 74 second-day", "none", "complete natural cohort", "VALID", rel(E2 / "plot_ready/panel_e_distance_from_return.csv"))
    add("ED2", "f", "quantitative simulation", "session", "B_SIM152", "23 naive + 58 return + 71 second-day", "valid missing-estimand restriction", "three 12-month second-day sessions lack the maintained half-coverage scalar", "VALID", rel(E2 / "plot_ready/panel_f_session_points.csv"))
    add("ED2", "g", "quantitative", "session", "B_DEPTH154", "22 naive + 58 return + 74 second-day", "valid limited support", "at least 10 events required in every depth stratum; 640day1 has 2 shallow events", "VALID", rel(E2 / "plot_ready/panel_g_supported_session_rows.csv"))
    add("ED2", "h", "quantitative", "session", "B_PB80", "22 naive + 58 return", "valid limited support", "same strict depth-support cohort as panel g", "VALID", rel(E2 / "plot_ready/panel_h_four_vs_depth_gap.csv"))
    add("ED2", "i", "quantitative simulation", "session", "B_SIM152", "23 naive + 58 return + 71 second-day", "valid missing-estimand restriction", "same scalar availability as panel f", "VALID", rel(E2 / "plot_ready/panel_i_session_points.csv"))

    # ED3
    for letter, cohort_id, breakdown, support in [
        ("a", "B_FLU_A63", "55 return + 8 latent", "468-9day1 and 769day2 have no eligible pre-reward explore-junction events"),
        ("b", "B_FLU_B65", "56 return + 9 latent", "all R56 and latent rows supported"),
        ("c", "B_FLU_C61", "53 return + 8 latent", "adds undefined direct-minus-other exclusions for 488-3day1 and 842-9day1"),
        ("d", "B_FLU_D65", "56 return + 9 latent", "all R56 and latent rows supported"),
    ]:
        add("ED3", letter, "quantitative", "session", cohort_id, breakdown, "valid fixed-month design and event support", support + "; R56 is intentionally the predefined fixed-month return cohort", "VALID", rel(E3 / "plot_ready/fluency_plot_rows.csv"))
    add("ED3", "e", "schematic", "none", "", "none", "not applicable", "route definition", "NOT_APPLICABLE", rel(E3 / "plot_ready/ed04_route_definition_current_plus_new12.csv"))
    add("ED3", "f", "quantitative descriptive", "session", "B_ROUTE162", "162 sessions", "valid fixed-month design restriction", "all six choices are resolved; the return stratum is intentionally the predefined fixed-month R56 cohort", "VALID", rel(E3 / "plot_ready/ed04_session_route_current_plus_new12.csv"))

    # ED4
    many("ED4", "a b", kind="illustrative example", statistical_unit="field of view / cell", cohort_ids="", breakdown="mouse 418 across four Day-1 sessions", exclusion_class="author-selected example", reason="not an inferential cohort", verdict="ILLUSTRATIVE_ONLY", source=rel(E4 / "embedded_sources"))
    add("ED4", "c", "quantitative", "animal trajectory", "N_C36", "14 animals", "longitudinal availability", "all clean36 animals contribute; displacement path length varies by follow-up", "VALID", rel(E4 / "source_from_raw/physical_fov_exact_36_source.csv"))
    many("ED4", "d e", kind="quantitative", statistical_unit="animal-cycle displacement", cohort_ids="N_C36", breakdown="13/12/7/4 animals at cycles 1/2/3/4", exclusion_class="valid longitudinal/design availability", reason="later cycle counts require later recordings; the 1616b 0->3 pair is off the modal month ladder", verdict="VALID", source=rel(E4 / "source_from_raw/physical_fov_exact_36_source.csv"))
    many("ED4", "f g", kind="quantitative", statistical_unit="directed candidate cell pair", cohort_ids="N_C36", breakdown="119,444 candidates from 36 session pairs / 14 animals", exclusion_class="none beyond clean36", reason="pre-registration candidate universe", verdict="VALID", source=rel(E4 / "source_from_raw/ed04_candidate_features_directed_long.csv"))
    many("ED4", "h i", kind="quantitative", statistical_unit="registered cell pair", cohort_ids="N_C36", breakdown="21,412 registered pairs from 36 session pairs / 14 animals", exclusion_class="none beyond clean36", reason="pre-activity-gate registration cohort", verdict="VALID", source=rel(E4 / "source_from_raw/ed04_registered_pair_features_long.csv"))

    # ED5
    add("ED5", "a", "illustrative gallery", "matched neuron", "", "32 author-selected examples", "qualified historical selection", "not sampled uniformly from the final cohort", "ILLUSTRATIVE_ONLY", rel(E5 / "plot_ready/selected_32.csv"))
    many("ED5", "b c d e", kind="quantitative", statistical_unit="long-gap pair / animal aggregate", cohort_ids="N_C36", breakdown="36 pairs / 14 animals", exclusion_class="valid clean36 design", reason="four day combinations or equal-time controls over the canonical cohort", verdict="VALID", source=rel(E5 / "plot_ready/ed06_day_combination_animal_values_corrected36.csv"))

    # ED6
    many("ED6", "a b", kind="schematic", statistical_unit="none", cohort_ids="", breakdown="none", exclusion_class="not applicable", reason="window definition diagrams", verdict="NOT_APPLICABLE", source=rel(E6 / "source_code/render_ed06_within_session_convergence_assembly.py"))
    many("ED6", "c e g", kind="quantitative curve", statistical_unit="animal-cycle pair", cohort_ids="N_SAME59", breakdown="59 cycles / 22 animals overall; time-bin membership decreases with duration", exclusion_class="valid per-bin time support", reason="curves stop or lighten according to frozen support masks", verdict="VALID", source=rel(E6 / "plot_ready/ed07_current59_curve_support_mask.csv"))
    many("ED6", "d f h", kind="quantitative curve", statistical_unit="animal-cycle pair", cohort_ids="N_SAME59", breakdown="59 cycles / 22 animals; each cycle uses its complete jointly available Day i/Day i+1 pivot range", exclusion_class="none beyond valid paired pivot support", reason="no minimum pivot-span exclusion; shortest retained curves contain 24 pivots from 30 to 53 min", verdict="VALID", source=rel(E6 / "source_data/available_curve_support_2026_08_16/ed06_pair_metric_available_curve_summaries.csv"))

    # ED7
    many("ED7", "a b c", kind="quantitative age model", statistical_unit="animal-cycle pair", cohort_ids="N_SAME58", breakdown="58 cycles / 21 animals", exclusion_class="valid metadata restriction", reason="animal 841 lacks date of birth required for age", verdict="VALID", source=rel(E7 / "plot_ready/ED07_FIRST30_PLOT_READY_MAINTAINED58.csv"))
    many("ED7", "d e f", kind="quantitative age model", statistical_unit="long-gap session pair", cohort_ids="N_C36", breakdown="36 pairs / 14 animals", exclusion_class="valid clean36 plus age-join completeness", reason="all clean36 pairs have required age fields", verdict="VALID", source=rel(E7 / "plot_ready/ed09_long_gap_age_joined_corrected36.csv"))

    # ED8
    add("ED8", "a", "quantitative map", "animal then spatial bin", "N_R30", "30 pairs / 14 animals; 104 bins supported in every animal", "valid common-bin and reward-context restriction", "equal-animal map requires bins supported by all 14 animals", "VALID", rel(E8 / "plot_ready/ed08_panel_h_map.csv"))
    add("ED8", "b", "quantitative", "long-gap session pair", "N_E8B_D1_30;N_E8B_D2_36", "Day i: 30 pairs / 14 animals; Day i+1: 36 pairs / 14 animals", "valid reward context and balanced-output support", "Day i requires rewarded-source eligibility; all finite valid balanced outputs are retained without a minimum source-exposure threshold", "VALID", rel(E8 / "source_data/ED08_PANEL_B_ALL_ELIGIBLE_PAIR_VALUES.csv"))
    add("ED8", "c", "quantitative", "long-gap session pair", "N_E8C_D1_30;N_E8C_D2_36", "D1 30 pairs / 14 animals; D2 36 pairs / 14 animals", "valid reward-context restriction", "D1 excludes six unrewarded source templates; D2 uses clean36", "VALID", rel(E8 / "plot_ready/ed08_panel_c_pair_values.csv"))
    add("ED8", "d", "quantitative", "long-gap session pair", "N_ROUTE_H29", "29 pairs / 13 animals", "valid run support; ledger gap", "requires >=5 runs in both sessions; upstream zero-run rows are not explicitly frozen", "VALID_BUT_STRENGTHEN_LEDGER", rel(E8 / "plot_ready/ed08_panel_d_pair_values.csv"), True)
    add("ED8", "e", "quantitative threshold curve", "long-gap session pair", "N_ROUTE_H29;N_ROUTE_R28", "29 approach / 13 animals; 28 return / 14 animals", "valid run support; ledger gap", "same route support as panel d; zero-run omissions should be explicit", "VALID_BUT_STRENGTHEN_LEDGER", rel(E8 / "plot_ready/ed08_panel_f_threshold_values.csv"), True)
    add("ED8", "f", "quantitative resampling control", "long-gap session pair", "N_ROUTE_H29", "29 pairs / 13 animals", "valid run/frame support", "same approach-route cohort with equal-frame resampling", "VALID", rel(E8 / "plot_ready/ed08_panel_e_pair_values.csv"))
    add("ED8", "g", "schematic", "none", "", "none", "not applicable", "route coordinates", "NOT_APPLICABLE", rel(E8 / "plot_ready/ed08_panel_a_routes.csv"))
    add("ED8", "h", "quantitative", "long-gap session pair", "N_ROUTE_2_30;N_ROUTE_3_29", "30 node-2 pairs / 14 animals; 29 node-3 pairs / 14 animals", "valid reward-context and run support", "requires a rewarded Day-i source and >=5 runs in both sessions", "VALID", rel(E8 / "plot_ready/ed08_panel_alt_pair_values.csv"))
    add("ED8", "i", "quantitative", "long-gap session pair", "N_OUT25", "25 pairs / 12 animals", "valid outcome support; ledger gap", "both rewarded and timeout outcomes required; explicit zero-outcome ledger is incomplete", "VALID_BUT_STRENGTHEN_LEDGER", rel(E8 / "plot_ready/ed08_panel_g_pair_values.csv"), True)

    # ED9
    many("ED9", "a b c e g h", kind="quantitative", statistical_unit="long-gap pair / nested neuron", cohort_ids="N_C36", breakdown="36 pairs / 14 animals; 16,901 matched neurons where applicable", exclusion_class="valid clean36/activity restriction", reason="canonical cohort with no manual neuron masks", verdict="VALID", source=rel(E9 / "SOURCE_INVENTORY.csv"))
    add("ED9", "d", "schematic", "none", "", "none", "not applicable", "partition diagram", "NOT_APPLICABLE", rel(E9 / "individual_panels/ED9_A1_tree_schematic.svg"))
    add("ED9", "f", "quantitative", "long-gap session pair", "N_C32", "32 pairs / 14 animals", "unjustified design restriction", "four support-valid 9->12-month pairs are labelled descriptive_9to12 and omitted from inference", "JUSTIFY_OR_ADD_SENSITIVITY", rel(E9 / "source_data/panel_f_inference_round24/be_roles/B2_EXCLUSION_CLEAN32_PAIR_CONTRASTS.csv"), True)
    add("ED9", "i", "quantitative curve", "session pair", "N_GAP47;N_C36", "47 at 1 day; 36/22/11/4 at longer gaps", "design plus longitudinal availability", "same gap-curve authority as Figure 2", "VALID_WITH_DOCUMENTED_SCOPE", rel(E9 / "plot_ready/F1_NO_MANUAL_MASK_GAP_CURVE.csv"))
    add("ED9", "j", "quantitative", "three-session chain / nested neuron", "N_CHAIN22", "22 chains / 11 animals", "valid design/support restriction", "requires two consecutive clean long gaps", "VALID", rel(E9 / "source_history/round4_powerpoint_parity/source_data/G_MAXIMAL26_POINTS.csv"))
    add("ED9", "k", "quantitative curve", "animal-cycle pair", "N_SAME58", "58 cycles / 21 animals", "unexplained inherited metadata restriction", "animal 841 was removed for missing DOB in ED7 age models, but ED9k does not use age and the 59th level-map row exists", "RESTORE_841_OR_JUSTIFY", rel(E9 / "source_history/round4_powerpoint_parity/source_data/H_STANDARD66_PLOT_READY.csv"), True)
    add("ED9", "l", "quantitative", "successive-cycle session pair", "N_S37", "37 pairs / 14 animals", "valid cycle-dependent scope", "includes one disclosed 14-day pair because the analysis is indexed by recording cycle rather than fixed month gap", "VALID", rel(E9 / "source_data/successive_cycle_bh4/main3_experience_trend_points_no_manual_mask.csv"))

    # ED10
    many("ED10", "a b", kind="quantitative", statistical_unit="long-gap session pair", cohort_ids="N_R30", breakdown="30 pairs / 14 animals; fixed K=166", exclusion_class="valid reward-context restriction", reason="source Day i must have rewarded experience; omission reason should also be copied into ED10's local membership table", verdict="VALID_BUT_STRENGTHEN_LEDGER", source=rel(E10 / "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv"), action=True)
    add("ED10", "c", "quantitative", "long-gap session pair", "N_C36", "36 pairs / 14 animals", "none beyond clean36", "all 36 pairs have supported policy values", "VALID", rel(E10 / "plot_ready/level_policy_bridge_join_corrected36.csv"))

    return panels


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = build_cohorts()
    panels = build_panels()

    expected = {
        "Figure 1": 8, "Figure 2": 17, "Figure 3": 13, "Figure 4": 6, "Figure 5": 6,
        "ED1": 11, "ED2": 9, "ED3": 6, "ED4": 9, "ED5": 5,
        "ED6": 8, "ED7": 6, "ED8": 9, "ED9": 12, "ED10": 3,
    }
    actual = {figure: sum(p.figure == figure for p in panels) for figure in expected}
    if actual != expected or len(panels) != 128:
        raise AssertionError(f"Panel enumeration mismatch: expected={expected}, actual={actual}, total={len(panels)}")
    figure_order = {figure: index for index, figure in enumerate(expected)}
    panels.sort(key=lambda p: (figure_order[p.figure], ord(p.panel)))

    assertions = {
        "B_N23": 23, "B_R56": 56, "B_R58": 58, "B_D74": 74, "B_L9": 9,
        "B_ALL153": 153, "B_EFF152": 152, "B_H153": 153, "B_PN22": 22,
        "B_SIM152": 152, "B_DEPTH154": 154, "N_C36": 36, "N_R30": 30,
        "N_SAME58": 58, "N_SAME59": 59, "N_PIV56": 56, "N_S37": 37,
        "N_CHAIN22": 22, "N_GAP47": 47, "N_C32": 32, "N_E8B_D1_30": 30,
        "N_E8B_D2_36": 36, "N_E8C_D1_30": 30, "N_E8C_D2_36": 36,
        "N_ROUTE_H29": 29, "N_ROUTE_R28": 28, "N_ROUTE_2_30": 30,
        "N_ROUTE_3_29": 29, "N_OUT25": 25,
    }
    for cohort_id, expected_n in assertions.items():
        actual_n = len(cohorts[cohort_id].units)
        if actual_n != expected_n:
            raise AssertionError(f"{cohort_id}: expected {expected_n}, got {actual_n}")

    # Exact discrepancy assertions that guard the main findings.
    r56 = {u.unit_id for u in cohorts["B_R56"].units}
    r58 = {u.unit_id for u in cohorts["B_R58"].units}
    if r58 - r56 != {"1616bday1", "1616b-3day1"} or r56 - r58:
        raise AssertionError("R56/R58 discrepancy changed")
    if {u.unit_id for u in cohorts["B_H153"].units} != {u.unit_id for u in cohorts["B_ALL153"].units}:
        raise AssertionError("ED1h matched-153 registry no longer equals the ED1 behavioral registry")
    if "640day1" in {u.unit_id for u in cohorts["B_PN22"].units}:
        raise AssertionError("640day1 unexpectedly passed the strict joint gate")
    if set(cohorts["N_C36"].animals) != set(cohorts["N_C32"].animals):
        raise AssertionError("ED9f should retain all 14 clean36 animals")
    c36 = {u.unit_id.replace("__", "->") for u in cohorts["N_C36"].units}
    c32 = {u.unit_id.replace("__", "->") for u in cohorts["N_C32"].units}
    missing_c32 = sorted(c36 - c32)
    expected_missing_c32 = sorted([
        "468-9day1->468-12day1", "837-9day1->837-12day1",
        "838-9day1->838-12day1", "839-9day1->839-12day1",
    ])
    if missing_c32 != expected_missing_c32:
        raise AssertionError(f"ED9f clean32 discrepancy changed: {missing_c32}")

    summary_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for panel in panels:
        cohort_ids = [c for c in panel.cohort_ids.split(";") if c]
        units: dict[str, Unit] = {}
        sources: list[str] = []
        for cohort_id in cohort_ids:
            cohort = cohorts[cohort_id]
            sources.append(cohort.source)
            for unit in cohort.units:
                units[f"{cohort_id}|{unit.unit_id}"] = unit
                unit_rows.append({
                    "figure": panel.figure,
                    "panel": panel.panel,
                    "cohort_id": cohort_id,
                    "unit_id": unit.unit_id,
                    "animal_id": unit.animal_id,
                    "status": "included",
                    "reason": cohort.restriction or "included by panel cohort definition",
                    "source": cohort.source,
                })
        animal_ids = sorted({u.animal_id for u in units.values()}, key=animal_sort_key)
        summary_rows.append({
            "figure": panel.figure,
            "panel": panel.panel,
            "panel_kind": panel.kind,
            "statistical_unit": panel.statistical_unit,
            "cohort_ids": panel.cohort_ids,
            "cohort_breakdown": panel.cohort_breakdown,
            "n_listed_units_across_cohorts": len(units) if cohort_ids else "",
            "n_unique_animals": len(animal_ids) if cohort_ids else "",
            "animal_ids": "|".join(animal_ids),
            "exclusion_class": panel.exclusion_class,
            "exclusion_reason": panel.exclusion_reason,
            "verdict": panel.verdict,
            "action_required": panel.action_required,
            "panel_source": panel.source,
            "cohort_sources": "|".join(dict.fromkeys(sources)),
            "note": panel.note,
        })

    write_csv(
        OUT / "panel_cohort_summary.csv", summary_rows,
        ["figure", "panel", "panel_kind", "statistical_unit", "cohort_ids", "cohort_breakdown",
         "n_listed_units_across_cohorts", "n_unique_animals", "animal_ids", "exclusion_class",
         "exclusion_reason", "verdict", "action_required", "panel_source", "cohort_sources", "note"]
    )
    write_csv(
        OUT / "panel_unit_ledger.csv", unit_rows,
        ["figure", "panel", "cohort_id", "unit_id", "animal_id", "status", "reason", "source"]
    )
    cohort_rows: list[dict[str, object]] = []
    for cohort in sorted(cohorts.values(), key=lambda c: c.cohort_id):
        cohort_rows.append({
            "cohort_id": cohort.cohort_id,
            "label": cohort.label,
            "n_units": len(cohort.units),
            "n_animals": len(cohort.animals),
            "animal_ids": "|".join(cohort.animals),
            "unit_ids": "|".join(u.unit_id for u in cohort.units),
            "restriction_or_caveat": cohort.restriction,
            "source": cohort.source,
        })
    write_csv(
        OUT / "cohort_inventory.csv", cohort_rows,
        ["cohort_id", "label", "n_units", "n_animals", "animal_ids", "unit_ids", "restriction_or_caveat", "source"]
    )

    # Explicit excluded/mismatched rows for fast review.
    findings: list[dict[str, object]] = [
        {"priority": "high", "panels": "ED2b,d", "unit_ids": "640day1",
         "classification": "caption/common-cohort mismatch", "reason": "all four policies are estimable; only shallow depth support fails (2<10)",
         "required_action": "include for the four-policy estimand or state that a stricter joint depth cohort was imposed"},
        {"priority": "high", "panels": "ED9f", "unit_ids": "|".join(expected_missing_c32),
         "classification": "unjustified design restriction", "reason": "all four 9->12-month pairs pass technical support and are merely tagged descriptive_9to12",
         "required_action": "provide scientific justification or add clean36 sensitivity/inference"},
        {"priority": "high", "panels": "ED9k", "unit_ids": "841day1->841day2",
         "classification": "irrelevant inherited exclusion", "reason": "missing DOB matters to ED7 age models but ED9k does not use age; the level-map row exists",
         "required_action": "restore animal 841 or document another panel-specific exclusion"},
        {"priority": "medium", "panels": "Figure4e", "unit_ids": "29 approach and 28 return pairs",
         "classification": "provenance gap", "reason": "active record contains a frozen SVG but no direct unit-level source table",
         "required_action": "copy the route unit ledger into the active Figure4 record"},
        {"priority": "medium", "panels": "ED8d,e,h,i", "unit_ids": "zero-run or zero-outcome candidate pairs",
         "classification": "auditability gap", "reason": "support exclusions are scientifically valid, but some zero-support rows disappear upstream",
         "required_action": "freeze explicit zero-support exclusion rows"},
        {"priority": "low", "panels": "ED10a,b", "unit_ids": "six unrewarded-source 0->3 pairs",
         "classification": "local provenance gap", "reason": "restriction is valid but ED10 membership table does not state it",
         "required_action": "copy reward-context exclusion reason into ED10_COHORT_MEMBERSHIP.csv"},
    ]
    write_csv(
        OUT / "findings_requiring_action.csv", findings,
        ["priority", "panels", "unit_ids", "classification", "reason", "required_action"]
    )

    exclusion_rows: list[dict[str, object]] = []

    def add_exclusion(
        figure: str,
        panel_letters: str,
        unit: Unit,
        classification: str,
        reason: str,
        eligible_universe: str,
        source: str,
        status: str = "excluded",
    ) -> None:
        for panel_letter in panel_letters.split():
            exclusion_rows.append({
                "figure": figure,
                "panel": panel_letter,
                "unit_id": unit.unit_id,
                "animal_id": unit.animal_id,
                "status": status,
                "classification": classification,
                "reason": reason,
                "eligible_universe": eligible_universe,
                "source": source,
            })

    # Estimand-specific fixed-month versus pooled-return roster difference.
    r58_lookup = {u.unit_id: u for u in cohorts["B_R58"].units}
    for unit_id in sorted(r58 - r56):
        unit = r58_lookup[unit_id]
        reason = "special animal-1616 sequence is not assigned to the predefined fixed 3/6/9/12-month behavior groups; it remains eligible for ED2's pooled-return policy estimand"
        add_exclusion("Figure 1", "d e g h", unit, "valid fixed-month design restriction", reason, "B_R58 pooled-return sensitivity", cohorts["B_R58"].source)
        add_exclusion("ED1", "b c d e f g j k", unit, "valid fixed-month design restriction", reason, "B_R58 pooled-return sensitivity", cohorts["B_R58"].source)
        add_exclusion("ED3", "a b c d f", unit, "valid fixed-month design restriction", reason, "B_R58 pooled-return sensitivity", cohorts["B_R58"].source)

    # ED1 valid efficiency support.
    eff_missing = {u.unit_id: u for u in cohorts["B_ALL153"].units if u.unit_id not in {x.unit_id for x in cohorts["B_EFF152"].units}}
    for unit in eff_missing.values():
        add_exclusion("ED1", "j k", unit, "valid limited estimand support",
                      "session never reaches 32 unique endpoints, so 32/N32 is undefined", "B_ALL153", cohorts["B_EFF152"].source)

    # ED2 shared-depth and simulation-support exclusions.
    naive_lookup = {u.unit_id: u for u in cohorts["B_N23"].units}
    unit640 = naive_lookup["640day1"]
    add_exclusion("ED2", "b", unit640, "criterion mismatch",
                  "all four policies are estimable; excluded only by an unstated shallow-depth common-cohort gate (2<10)",
                  "B_N23", cohorts["B_PN22"].source)
    add_exclusion("ED2", "d", unit640, "inherited criterion mismatch",
                  "naive substitution mean inherits the strict depth gate although the simulated estimand uses four policies",
                  "B_N23", cohorts["B_PN22"].source)
    add_exclusion("ED2", "g h", unit640, "valid limited depth support",
                  "only 2 shallow-depth events; at least 10 required in every depth stratum", "B_N23", cohorts["B_DEPTH154"].source)
    sim_ids = {u.unit_id for u in cohorts["B_SIM152"].units}
    for unit in cohorts["B_D74"].units:
        if unit.unit_id not in sim_ids:
            add_exclusion("ED2", "f i", unit, "valid missing-estimand support",
                          "maintained half-coverage scalar is absent", "B_D74", cohorts["B_SIM152"].source)

    # ED3 event-defined omissions.
    behavior_by_id = {u.unit_id: u for c in (cohorts["B_R56"], cohorts["B_L9"]) for u in c.units}
    for unit_id in ("468-9day1", "769day2"):
        add_exclusion("ED3", "a c", behavior_by_id[unit_id], "valid event support",
                      "no eligible pre-first-reward exploration-junction events", "B_R56+B_L9", cohorts["B_FLU_A63"].source)
    for unit_id in ("488-3day1", "842-9day1"):
        add_exclusion("ED3", "c", behavior_by_id[unit_id], "valid contrast support",
                      "direct-path events exist but other-junction events do not, so direct-minus-other is undefined",
                      "B_R56+B_L9", cohorts["B_FLU_C61"].source)

    # Canonical long-gap and successive-cycle scope.
    short_pair = next(u for u in cohorts["N_S37"].units if u.unit_id == "1616day1->1616bday1")
    clean36_panels = {
        "Figure 2": "d e i j n o", "Figure 4": "b", "Figure 5": "b c d e f",
        "ED4": "c d e f g h i", "ED5": "b c d e", "ED7": "d e f",
        "ED9": "a b c e g h", "ED10": "c",
    }
    for figure, letters in clean36_panels.items():
        add_exclusion(figure, letters, short_pair, "valid primary long-gap design restriction",
                      "14-day interval is excluded from the canonical clean36 long-gap cohort", "N_S37", cohorts["N_C36"].source)

    # Reward-source eligibility in Figure4/ED8/ED10.
    reward_ids = {u.unit_id.replace("__", "->") for u in cohorts["N_R30"].units}
    for unit in cohorts["N_C36"].units:
        if unit.unit_id.replace("__", "->") not in reward_ids:
            reason = "source Day i is an unrewarded template, so reward-location decoding is not biologically defined"
            add_exclusion("Figure 4", "d f", unit, "valid reward-context restriction", reason, "N_C36", cohorts["N_R30"].source)
            add_exclusion("ED8", "a c", unit, "valid reward-context restriction", reason, "N_C36", cohorts["N_R30"].source)
            add_exclusion("ED10", "a b", unit, "valid reward-context restriction", reason, "N_C36", cohorts["N_R30"].source)

    # Same-stage support/metadata omissions.
    same59_lookup = {u.unit_id: u for u in cohorts["N_SAME59"].units}
    unit841 = same59_lookup["841day1->841day2"]
    add_exclusion("ED7", "a b c", unit841, "valid metadata restriction",
                  "date of birth is unavailable, so age cannot be computed", "N_SAME59", cohorts["N_SAME58"].source)
    add_exclusion("ED9", "k", unit841, "irrelevant inherited restriction",
                  "DOB is unavailable, but ED9k does not use age and a supported level-map row exists", "N_SAME59", cohorts["N_SAME58"].source)

    # ED9f supported 12-month pairs.
    clean36_lookup = {u.unit_id.replace("__", "->"): u for u in cohorts["N_C36"].units}
    for unit_id in expected_missing_c32:
        add_exclusion("ED9", "f", clean36_lookup[unit_id], "unjustified design restriction",
                      "technical support passes; upstream row is tagged descriptive_9to12 rather than inferential",
                      "N_C36", cohorts["N_C32"].source)

    # ED8 exact pair losses relative to clean36. Canonicalize the different ID separators.
    def canonical_pair(value: str) -> str:
        return value.replace("__", "->").replace("_to_", "->")

    ed8_specs = [
        ("b", "N_E8B_D1_30", "valid reward-context restriction", "Day-i source must have rewarded experience; source exposure is metadata, not a gate"),
        ("d", "N_ROUTE_H29", "valid route-run support", ">=5 eligible home-to-water runs required in both sessions"),
        ("e", "N_ROUTE_H29", "valid route-run support", ">=5 eligible home-to-water runs required in both sessions"),
        ("e", "N_ROUTE_R28", "valid route-run support", ">=5 eligible water-to-home runs required in both sessions"),
        ("f", "N_ROUTE_H29", "valid route-run/frame support", "route support followed by equal-frame resampling"),
        ("h", "N_ROUTE_2_30", "valid reward-context and route-run support", "rewarded Day-i source and >=5 eligible node-2-to-water runs in both sessions"),
        ("h", "N_ROUTE_3_29", "valid reward-context and route-run support", "rewarded Day-i source and >=5 eligible node-3-to-water runs in both sessions"),
        ("i", "N_OUT25", "valid outcome support", "both rewarded and timeout route outcomes are required"),
    ]
    for panel_letter, cohort_id, classification, reason in ed8_specs:
        included = {canonical_pair(u.unit_id) for u in cohorts[cohort_id].units}
        for baseline_id, unit in clean36_lookup.items():
            if baseline_id not in included:
                add_exclusion("ED8", panel_letter, unit, classification, reason,
                              "N_C36", cohorts[cohort_id].source)

    write_csv(
        OUT / "panel_exclusion_ledger.csv", exclusion_rows,
        ["figure", "panel", "unit_id", "animal_id", "status", "classification", "reason", "eligible_universe", "source"]
    )

    authority_files = {
        ROOT / "docs/figure_version_records/CURRENT_MAIN_FIGURES.md",
        ROOT / "docs/figure_version_records/CURRENT_EXTENDED_DATA_FIGURES.md",
        ROOT / "docs/manuscript_figure_captions_2026-08-10/figure_captions_main1-5_extended1-10.tex",
    }
    source_paths = sorted({ROOT / row["panel_source"] for row in summary_rows if row["panel_source"] and (ROOT / str(row["panel_source"])).is_file()}
                          | {ROOT / cohort.source for cohort in cohorts.values() if (ROOT / cohort.source).is_file()}
                          | authority_files)
    source_rows = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_paths]
    write_csv(OUT / "source_manifest.csv", source_rows, ["path", "bytes", "sha256"])

    by_figure: dict[str, list[Panel]] = {figure: [] for figure in expected}
    for panel in panels:
        by_figure[panel.figure].append(panel)
    action_panels = [p for p in panels if p.action_required]
    report: list[str] = [
        "# Manuscript-wide panel cohort and exclusion audit",
        "",
        "## Scope and frozen authority",
        "",
        "This is a read-only scientific audit of the 15 pointer-selected figure records as they stood on 2026-08-15. "
        "It enumerates all 128 captioned panels (50 main-figure and 78 Extended Data panels). Schematics and author-selected examples are separated from cohort-level analyses. Figures and captions were not changed.",
        "",
        "The panel table is an index; exact animal and session/pair IDs are in `cohort_inventory.csv` and are expanded panel-by-panel in `panel_unit_ledger.csv`.",
        "",
        "## Direct answer: why 22 naive animals instead of 23?",
        "",
        "The missing session is **640day1** (animal **640**). It has all four policies finite and adequately supported "
        "(BF=16, BA=16, SF/LF=30, BS/LO=30 eligible observations), but only **2 shallow-depth** leaving events. "
        "The strict shared gate requires at least 10 events in every depth stratum, so the exclusion is valid for ED2g/h. "
        "It is not valid under ED2b's captioned criterion, 'all four policies were estimable': 640day1 satisfies that criterion. "
        "ED2b inherited a stricter cross-panel depth gate that its caption does not disclose.",
        "",
        "## Correct interpretation of R56 versus R58",
        "",
        "R56 and R58 are different eligible universes, not evidence that Figure 1 omitted two valid fixed-month rows. "
        "R56 is the union of the predefined 3-, 6-, 9- and 12-month return groups (17/17/12/10 sessions). "
        "The special animal-1616 sequence, 1616bday1 and 1616b-3day1, is not cleanly assigned to that calendar-month behavior registry and is intentionally excluded. "
        "ED2 instead pools return sessions without requiring a fixed-month label, so both sessions are eligible there. "
        "Cycle-dependent analyses may also use the special sequence, and Figure 2 validly retains 1616bday1->1616b-3day1 in its non-month-stratified long-gap cohort while excluding the preceding 14-day pair from clean long-gap analyses.",
        "",
        "## Findings that require action",
        "",
        "1. **ED2b's caption states the wrong inclusion rule.** 640day1 is lost only through a shared depth-support gate; ED2d's naive substitution mean inherits the same unnecessary restriction for its four-policy estimand.",
        "2. **ED9f excludes four technically supported 9-to-12-month pairs.** They are tagged descriptive_9to12 rather than failing support; justify the restriction or show the clean36 sensitivity.",
        "3. **ED9k excludes animal 841 for a reason irrelevant to that panel.** Missing date of birth justifies ED7 age-model exclusion, but ED9k is not an age model and its 59th level-map row exists.",
        "4. **Figure4e lacks a direct unit-level source table in its active record.** The likely 29/28 pair cohorts can be reconstructed from ED8, but Figure4 should freeze its own ledger.",
        "",
        "## Valid recurring exclusions",
        "",
        "- Limited time/bin/run/outcome support: ED1j/k, ED2f-i, ED3a/c and ED8b-i. ED6d/f/h now retain all 59 cycles over pair-specific complete available pivot ranges.",
        "- Reward context: Figure4d/f, ED8a-c and ED10a/b exclude unrewarded source templates.",
        "- Longitudinal availability: Figure2/ED9 gap curves and ED4 later cycles.",
        "- Metadata: ED7a-c validly excludes animal 841 because age cannot be computed without DOB.",
        "- Fixed-month behavior design: Figure1, ED1 and ED3 use R56 because only sessions with unambiguous 3/6/9/12-month labels enter that registry; ED2 validly uses R58 for its pooled-return estimand.",
        "- Cycle-dependent versus gap-dependent design: clean36 excludes the 14-day 1616 pair, whereas Figure3g/j/m and ED9l validly include it under their disclosed recording-cycle estimand.",
        "",
        "## Panel-by-panel audit",
        "",
        "| Figure | Panel | Kind | Cohort / count | Verdict | Exclusion or caveat |",
        "|---|---:|---|---|---|---|",
    ]
    for panel in panels:
        report.append(
            f"| {panel.figure} | {panel.panel} | {panel.kind} | {panel.cohort_breakdown} | {panel.verdict} | {panel.exclusion_reason} |"
        )
    report += [
        "",
        "## Interpretation boundary",
        "",
        "There is no scientifically meaningful single denominator for all panels. The audit therefore compares every panel against its own eligible universe: behavior session, adjacent-day cycle, long-gap pair, route-supported pair, outcome-supported pair, or three-session chain. Behavior-only animals are not labelled as behaviorally excluded merely because they lack neural registration.",
        "",
        "## Files",
        "",
        "- `panel_cohort_summary.csv`: one row for every displayed panel.",
        "- `cohort_inventory.csv`: exact animals and exact session/pair IDs for each reused cohort.",
        "- `panel_unit_ledger.csv`: expanded panel-to-unit membership.",
        "- `panel_exclusion_ledger.csv`: exact excluded or extra units with panel-specific reasons.",
        "- `findings_requiring_action.csv`: exact discrepancies and required follow-up.",
        "- `source_manifest.csv`: SHA-256 closure for all directly read source tables.",
        "- `validation.json`: count assertions and audit limitations.",
        "",
        "## Validation and limitations",
        "",
        f"- Enumerated panels: {len(panels)} / 128.",
        f"- Panels marked for action or stronger provenance: {len(action_panels)}.",
        "- All declared cohort-count assertions passed.",
        "- `python scripts/validate_current_figure_records.py` is executed by the audit script; its current status and any failed record checks are preserved in `validation.json`.",
        "- Exact curve-bin membership is not repeated in this report; the frozen support-mask tables remain authoritative for time-varying support.",
        "- Figure4e is not independently unit-auditable from its active record because only a frozen SVG is present.",
        "- This audit did not rerun expensive raw calcium extraction, alter panels, or compile the LaTeX manuscript.",
    ]
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    record_validator = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_current_figure_records.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    record_validation_payload: dict[str, object] = {}
    try:
        record_validation_payload = json.loads(record_validator.stdout)
        record_validation_status = record_validation_payload.get("status", "unknown")
    except json.JSONDecodeError:
        record_validation_status = "unparseable"
    record_validation_pass = record_validator.returncode == 0 and record_validation_status == "pass"
    failed_record_checks = {
        name: sorted(check for check, passed in record.get("checks", {}).items() if not passed)
        for name, record in record_validation_payload.get("records", {}).items()
        if any(not passed for passed in record.get("checks", {}).values())
    }

    validation = {
        "status": (
            "PASS_WITH_ACTION_ITEMS"
            if record_validation_pass
            else "COHORT_AUDIT_PASS_WITH_EXTERNAL_RECORD_VALIDATOR_FAILURE"
        ),
        "generated": "2026-08-15",
        "backend": "Python standard library",
        "figure_records_frozen": 15,
        "panels_expected": 128,
        "panels_enumerated": len(panels),
        "figure_panel_counts_expected": expected,
        "figure_panel_counts_actual": actual,
        "cohort_count_assertions": assertions,
        "action_panel_count": len(action_panels),
        "explicit_exclusion_ledger_rows": len(exclusion_rows),
        "current_figure_record_validator": {
            "command": "python scripts/validate_current_figure_records.py",
            "status": record_validation_status,
            "figure_count": record_validation_payload.get("figure_count"),
            "panel_count": record_validation_payload.get("panel_count"),
            "failed_record_checks": failed_record_checks,
            "stderr": record_validator.stderr.strip(),
        },
        "critical_findings": [
            "ED2b caption criterion does not describe the implemented joint depth gate",
            "ED9f omits four support-valid 9-to-12-month pairs",
            "ED9k inherits an age-metadata exclusion although age is not its estimand",
            "Figure4e lacks a direct unit-level source table",
        ],
        "not_run": ["raw calcium extraction", "figure regeneration", "LaTeX compilation"],
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": validation["status"], "panels": len(panels), "cohorts": len(cohorts), "output": rel(OUT)}, indent=2))


if __name__ == "__main__":
    main()
