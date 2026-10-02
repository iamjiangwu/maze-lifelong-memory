# Extended Data Figure 10: data

Matched neural-behavior observations and their inferential results. [Column and format index](index.csv).

| Panel | Inputs and observation unit |
|---|---|
| a | `neural_behavior_pair_rows.csv`: one matched neural/return-behavior session pair with six neural readouts and three behaviors. `neural_behavior_correlation_statistics.csv` contains descriptive correlations; `panel_a_expected_multiplicity.csv` supplies regression p values and row-wise BH3 stars. |
| b | The same pair rows plus `reward_level_commonality.csv` (explained-variance components) and `reward_level_profile_tests.csv` (the three behavioral-profile comparisons). |
| c | `level_policy_bridge_join.csv`: one matched pair's level-map correlation and four first-30-min turning biases; `level_policy_bridge_stats.csv` contains fits and BH4 p values. |

Raw reward/whole-maze decoding errors are spatial bins, first long bout time is seconds, and raw correlations/efficiency ratios are dimensionless. Columns such as `reward_location`, `whole_decoder`, `structure`, `earlier_first_long` and `exploration_efficiency` are the transformed/standardized model readouts, distinct from the corresponding raw fields. `log_reward_over_pooled` is the log reward-approach-speed ratio to pooled controls. Reward decoder readouts are oriented so better decoding is higher; earlier-first-long-bout readouts are oriented so earlier reward acquisition is higher. Support fields record common cells/bins and technical subsamples; subsamples do not create independent observations.

Panel a displays absolute Pearson r, but its stars test slopes with standard errors clustered by animal and BH3 over the three behaviors in each neural row. Panel b's `unique_a`, `shared`, `unique_b` partition R² for reward decoding and level maps; these bars are not the profile tests. The profile estimands are reward-minus-level association for earlier first long bout, reward-minus-level association for exploration, and the difference between those two contrasts. Their exact animal-score sign-flip p values form BH3.

In panel c, `Bf`, `Ba`, `Lf`, `Lo_depth_weighted` correspond to P_SF, P_SA, P_BF and depth-weighted P_BS; they are probabilities, not separate behavioral windows. `policy_supported` records eligibility. See [statistical definitions](../../../docs/STATISTICS.md) for the correction families.

| File | Format |
|---|---|
| [level_policy_bridge_join.csv](level_policy_bridge_join.csv) | .csv |
| [level_policy_bridge_stats.csv](level_policy_bridge_stats.csv) | .csv |
| [neural_behavior_correlation_statistics.csv](neural_behavior_correlation_statistics.csv) | .csv |
| [neural_behavior_pair_rows.csv](neural_behavior_pair_rows.csv) | .csv |
| [panel_a_expected_multiplicity.csv](panel_a_expected_multiplicity.csv) | .csv |
| [reward_level_commonality.csv](reward_level_commonality.csv) | .csv |
| [reward_level_profile_tests.csv](reward_level_profile_tests.csv) | .csv |
