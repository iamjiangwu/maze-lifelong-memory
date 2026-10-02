# Figure 2: data

Long-gap neural correlations, saved neuron-ID null distributions and example neurons. [Column and array index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–c, h, m | `*_corr.npz` supplies example local-correlation images; the four neuron-example NPZs supply registered-neuron trajectories and signals used in the examples. Representation schematics are drawn by code. |
| d, i, n | `*_distribution.npz`: PV correlations per spatial bin in d, place-map correlations per matched neuron in i, and ensemble-rate correlations per session pair in n, with saved shuffle histograms/values. |
| e, j, o | `*_shuffle_mean.npz`: null draws of the cohort mean and the observed mean; `pair_metric_summary.csv` and `metric_summary.csv` give the pair/cohort summaries. |
| f, k, p | `gap_with_day1_day2_curve_data.csv` contains means and 95% intervals by Days apart. `same_stage_*`, `long_gap_pair_rows.csv` and the ensemble-rate pair tables supply the corresponding observations. |
| g, l, q | `consecutive_gap_points.csv` contains one preceding/following gap comparison per spatial bin within a three-session chain for g, and per matched neuron within a chain for l. The canonical ensemble-rate consecutive-gap table has one observation per chain for q. Statistics and `displayed_estimand_bh4.csv` supply the fitted tests, with standard errors clustered by animal. |

All correlation values are dimensionless. Distribution observations have different units from cohort means: the latter equally weight session-pair means. `n_pairs_final`, finite neuron/bin counts and `distribution_pair_support.csv` record support; `consecutive_pair_scope.csv` records eligible triplets. BH4 also includes the displayed level-map association in ED9l.

In `distribution_pair_support.csv`, `n_pairs_final` counts matched-neuron pairs within a session pair. `n_pairs_dropped_by_user` counts manually excluded matched-neuron pairs; every supplied row is zero. Neither field counts mice or recording-session pairs.

In neuron NPZs, A/B identify source/target sessions; `*_frame` indexes the saved signal trace, while the shorter coordinate arrays contain retained position samples. Do not align these by row number without `*_frame`. Positions are normalized maze coordinates (approximately 0–1), `dt` is seconds per frame, `Craw` is the calcium trace and `S` the inferred activity signal. `place_r` is the 15 × 15 place-map correlation and `place_support_15` its spatial support count. Cell identifiers are indices within the named sessions, not animal-wide neuron IDs.

| File | Format |
|---|---|
| [418-3day1_to_418-6day1__row0062_a83_b794.npz](418-3day1_to_418-6day1__row0062_a83_b794.npz) | .npz |
| [422-3_corr.npz](422-3_corr.npz) | Local-correlation image, coordinates and color metadata |
| [422_corr.npz](422_corr.npz) | Local-correlation image, coordinates and color metadata |
| [451-6day1_to_451-9day1__row0305_a402_b408.npz](451-6day1_to_451-9day1__row0305_a402_b408.npz) | .npz |
| [837-6day1_to_837-9day1__row0175_a187_b131.npz](837-6day1_to_837-9day1__row0175_a187_b131.npz) | .npz |
| [837-6day1_to_837-9day1__row0588_a674_b683.npz](837-6day1_to_837-9day1__row0588_a674_b683.npz) | .npz |
| [adjacent_day_ensemble_rate_pair_values.csv](adjacent_day_ensemble_rate_pair_values.csv) | .csv |
| [canonical_ensemble_rate_all_gap_pair_values.csv](canonical_ensemble_rate_all_gap_pair_values.csv) | .csv |
| [consecutive_gap_points.csv](consecutive_gap_points.csv) | .csv |
| [consecutive_gap_statistics.csv](consecutive_gap_statistics.csv) | .csv |
| [consecutive_pair_scope.csv](consecutive_pair_scope.csv) | .csv |
| [displayed_estimand_bh4.csv](displayed_estimand_bh4.csv) | .csv |
| [distribution_pair_support.csv](distribution_pair_support.csv) | .csv |
| [ensemble_rate_distribution.npz](ensemble_rate_distribution.npz) | .npz |
| [ensemble_rate_shuffle_mean.npz](ensemble_rate_shuffle_mean.npz) | .npz |
| [gap_with_day1_day2_curve_data.csv](gap_with_day1_day2_curve_data.csv) | .csv |
| [long_gap_pair_rows.csv](long_gap_pair_rows.csv) | .csv |
| [metric_summary.csv](metric_summary.csv) | .csv |
| [pair_metric_summary.csv](pair_metric_summary.csv) | .csv |
| [place_map_distribution.npz](place_map_distribution.npz) | .npz |
| [place_map_shuffle_mean.npz](place_map_shuffle_mean.npz) | .npz |
| [population_vector_distribution.npz](population_vector_distribution.npz) | .npz |
| [population_vector_shuffle_mean.npz](population_vector_shuffle_mean.npz) | .npz |
| [same_stage_correlations.csv](same_stage_correlations.csv) | .csv |
| [same_stage_shuffle.csv](same_stage_shuffle.csv) | .csv |
| [session_pair_rate_consecutive_gap_points_canonical.csv](session_pair_rate_consecutive_gap_points_canonical.csv) | .csv |
| [session_pair_rate_consecutive_gap_stats_canonical.csv](session_pair_rate_consecutive_gap_stats_canonical.csv) | .csv |
