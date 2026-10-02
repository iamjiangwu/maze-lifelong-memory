# Extended Data Figure 5: data

Registered-neuron examples and session-pair comparisons across the long gap. [Column and array index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a | `selected_maps_fields_45_shortlist32.npz`, `selected_32.csv` and `ed5_panel_a_*`: 32 neuron examples, their displayed maps, identities and correlation labels. |
| b–d | `place_`, `pv_` and `rate_pair_summary_all_variants.csv`: one session pair × day combination per neural metric. The builder combines these measurements into a long-format table; descriptive and contrast CSVs supply the resulting means, intervals and tests. |
| e | `ed05e_context_only_all_outside_pair_rows.csv` and `ed05e_context_only_all_outside_clustered_stats.csv`: equal-sample maze-versus-outside ensemble-rate comparisons for Day i versus Day i+90. The latter's `maze` and `outside` rows supply condition means; `maze_minus_outside` supplies their contrast. |

Day codes `d1–d1`, `d1–d2`, `d2–d1`, `d2–d2` mean Day i/i+90, i/i+91, i+1/i+90 and i+1/i+91. Correlations are dimensionless. Panel-a maps are displayed at 45 × 45 resolution, but their labels use the **15 × 15 place-map correlation** (`analysis_r15`/`place_r_15_analysis`) matching Figure 2. The `common_mask` denotes display support, not a separate correlation estimator.

The three pair tables use `place_map_mean_all`, `data_mean` and `pearson_raw_r` for place-map, PV and ensemble-rate correlation, respectively. Session and animal identifiers link the four day combinations; matched-neuron counts describe support, not the weight assigned to a pair.

`n_pairs_dropped_by_user` in the PV table and `manual_user_drops` in the exact-support table count manually excluded matched-neuron pairs, not session pairs. All values are zero; the builder checks this and verifies that the three neural measures use the same matched-neuron support.

Panels b–d equally weight session-pair values for cohort means. Confidence intervals use standard errors clustered by animal; the three contrasts within each metric use exact sign-flips with every pair of an animal flipped together, then BH3. Their plotted p values are `p_bh3_within_metric_exact_cluster_signflip`, not the separately recorded regression p values. `feasible` and frame counts describe equal-sampling support in panel e; only the displayed day combination enters its test.

| File | Format |
|---|---|
| [day_combination_session_pair_contrasts_cr1_bh3.csv](day_combination_session_pair_contrasts_cr1_bh3.csv) | .csv |
| [day_combination_session_pair_descriptive_cr1.csv](day_combination_session_pair_descriptive_cr1.csv) | .csv |
| [ed05e_context_only_all_outside_clustered_stats.csv](ed05e_context_only_all_outside_clustered_stats.csv) | .csv |
| [ed05e_context_only_all_outside_pair_rows.csv](ed05e_context_only_all_outside_pair_rows.csv) | .csv |
| [ed5_panel_a_per_side_provenance.csv](ed5_panel_a_per_side_provenance.csv) | .csv |
| [ed5_panel_a_provenance_summary.json](ed5_panel_a_provenance_summary.json) | .json |
| [place_pair_summary_all_variants.csv](place_pair_summary_all_variants.csv) | .csv |
| [place_pv_rate_exact_support_audit.csv](place_pv_rate_exact_support_audit.csv) | .csv |
| [pv_pair_summary_all_variants.csv](pv_pair_summary_all_variants.csv) | .csv |
| [rate_pair_summary_all_variants.csv](rate_pair_summary_all_variants.csv) | .csv |
| [selected_32.csv](selected_32.csv) | .csv |
| [selected_maps_fields_45_shortlist32.npz](selected_maps_fields_45_shortlist32.npz) | .npz |
