# Statistical replay

Run `python reproduce.py --statistics-only --output-dir outputs/statistics` from the repository root. This recalculates all 53 recorded analysis groups from included derived observations and saved null/simulation draws. Figures use the frozen plotting inputs; the separate replay checks their statistical results. Map fitting, decoder fitting and raw-data processing remain upstream.

Correction families and statistical units are preserved, not reselected during repository cleanup. In particular, ED10a uses mouse-clustered CR1 with BH3 across each neural-measure row; Figure 4’s matching regression uses nominal CR1 p-values. ED5 preserves its session-pair means and mouse-clustered contrasts.

## Read the outputs

The chosen output folder contains `statistics/STATISTICS_REPORT.json`, regenerated numerical tables, verification tables and `BUILD.log`. The report lists each group’s status and links it to its regenerated table. Detailed units/families are recorded by the analysis implementations and the original executed report below; do not treat distinct correction families as one global family.

## Find an analysis

| Analysis group | Recorded unit/family or implementation |
|---|---|
| `main1_reward_acquisition_day1_month` | session observations; CR1 clustered by mouse; log(lb0) response; four naive-versus-return-month contrasts BH4 |
| `main1_long_bout_group` | session observations; CR1 clustered by mouse; log(lb0) response; three pairwise reward-context contrasts BH3 |
| `main1_route_contrasts` | 162 sessions; four-level saturated condition model; CR1 clustered by 33 mice; six route-score pairwise contrasts BH6 |
| `main1_route_group_ci` | condition means from same saturated CR1 model; none; 95% marginal t(G-1) intervals |
| `main1_reward_acquisition_day1_month_summary` | session-level descriptive mean/SEM; none |
| `main1_long_bout_group_summary` | session-level descriptive mean/SEM; none |
| `main2_dense_previous_next` | neuron/bin observations; CR1 clustered by mouse; historical dense four-metric BH4 context, not current displayed ensemble-rate estimand |
| `main2_rate_previous_next` | 22 session chains, 11 mice; CR1; current Figure2 displayed BH4 recomputed below |
| `main2_displayed_previous_next_bh4` | metric-specific neuron/bin or session-chain observations, clustered by mouse; three dense metrics plus canonical pair-level ensemble rate BH4 |
| `main2_shuffle_mean_tests` | equal pair means; supplied 10,000 technical null draws; one-sided plus-one Monte Carlo; none |
| `main2_gap_curve_pair_mean_ci` | equal session-pair means; t(N_pairs-1) descriptive 95% intervals; none; no gap-wise hypothesis tests used by panel |
| `main3_first30_experience_trends` | pair/animal–experience rows; CR1 clustered by animal; separate four-representation BH4 family; decoder sensitivities excluded |
| `main3_long_gap_experience_trends` | pair/animal–experience rows; CR1 clustered by animal; separate four-representation BH4 family; decoder sensitivities excluded |
| `main3_day2_template_curve_mean_sem` | pairs averaged within animal in each source cohort; pooled cohort animal mean/variance; none; mean ± SEM descriptive curves |
| `main4_decoder_advantages` | within-pair advantages; intercept-only OLS with mouse-clustered CR1; none; prespecified nominal tests |
| `main4_reward_behavior_k166` | 30 pair means from 14 mice; behavior ~ standardized reward decoding error; CR1; none; raw p, unlike ED10a BH3 |
| `main4_route_animal_profiles` | pairs within mouse, then equally weighted animals; none |
| `main4_route_enrichment` | animal-mean effects; exact two-sided sign-flip of mean (all 2^G signs); 95% t intervals; two direction enrichment tests BH2; direction difference separate |
| `main4_route_curve` | equal animals; t(G-1) descriptive 95% intervals; none |
| `main5_level_exploration` | 36 session-pair observations, CR1 clustered by 14 mice; none; prespecified nominal association |
| `main5_level_pv_matrix` | cell-wise equal session-pair means, 32 or 36 supported pairs; none; descriptive heatmap |
| `ed1_endpoint_bh27` | eligible session rows, log1p outcomes, CR1 clustered by canonical biological animal; 27 endpoint×axis×group contrasts BH27 |
| `ed1_tile_time` | 153 sessions; log-log OLS, canonical mouse-clustered CR1; none; single prespecified diagnostic |
| `ed1_speed_mean_sem` | session speeds, first 30 min in-maze, 60 cm/s segment cap; mean/SEM; none; descriptive only |
| `ed2_policy` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed2_depth` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed2_matched_policy_contributions` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed2_matched_gap` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed3_route_bh6` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed3_fluency_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed5_day_combination_means` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed5_day_combination_contrasts` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed5_context_control_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed7_first30_age_bh3` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed7_long_gap_age_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_level_gap_pair_means` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_decoder` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_axis_controls_bh2` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_within_level_control` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_first30_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_cycles_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_consecutive_bh4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ed9_population_vector_null_summary` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED6_fixed_reference_CR1_BH18` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED6_pivot_CR1_BH9` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED8b_equal_mouse_exact_BH4` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED8c_pair_CR1_BH2` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED8_standardized_route_exact_signflip_and_BH` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED10a_Pearson_CR1_BH3_and_BH6` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED10a_CR1_nominal_and_adjusted_p_parity` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED10b_commonality` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED10b_CR1_exact_cluster_score_maxT3` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |
| `ED10c_policy_CR1_and_window_BH18` | See the implementation and [executed report](validation/STATISTICS_REPORT.json). |

[Original executed statistical report](validation/STATISTICS_REPORT.json) · [Reproduction instructions](REPRODUCING.md)
