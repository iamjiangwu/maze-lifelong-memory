# Extended Data Figure 7: data

Age associations conditioned on recording experience. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–c | `first30_plot_ready.csv`: one animal/recording-cycle × metric's first-30-min alignment to its Day-2 reference, with age and cycle index. `first30_age_models.csv` contains fitted age coefficients. |
| d–f | `long_gap_age_joined.csv`: one long-gap session pair × metric with source/target ages and transition. `long_gap_age_transition_fixed_models.csv` contains the transition-adjusted fits. |

`age_day1_days`, `age_src_days` and `age_dst_days` are calendar age in days; `experience_index` and `transition` are recording-experience categories, not age. The upper panels use Day-1 age, the lower panels destination-session age. `value` is a dimensionless neural correlation. `beta_per_30d` is the correlation change per 30 days of age after controlling for experience/transition.

The upper panels test age associations in three early-alignment measures—place maps, PVs and ensemble rates—with BH3. The lower panels belong to the four-measure long-gap age analysis, which adds level maps and uses BH4. No level-map early-alignment age model is supplied. These are separate families with different scopes, not a common six-panel correction.

The fitted models weight the included observations equally and use standard errors clustered by animal; they do not first average all of an animal's pairs into one point. `age_control_included`, exclusions and model specification fields record eligibility.

The fourth lower-family member, the level-map age association, is tabulated in `long_gap_age_transition_fixed_models.csv` but not plotted in ED7: slope −0.00102 per 30 days, raw p = 0.894, BH4-adjusted p = 0.894. It remains in BH4; the ensemble-rate adjusted p is 0.092. This differs from Figure 3/ED9n's early-alignment **experience** family, which includes level maps; age and experience tests are not interchangeable.

`source_bundles` identifies sorted-neuron cohorts or original-neuron naïve fallbacks; it is a label for readers, not a renderer input. See [source-label definitions](../../../docs/DATA.md#source-labels).

| File | Format |
|---|---|
| [first30_age_models.csv](first30_age_models.csv) | .csv |
| [first30_plot_ready.csv](first30_plot_ready.csv) | .csv |
| [long_gap_age_joined.csv](long_gap_age_joined.csv) | .csv |
| [long_gap_age_transition_fixed_models.csv](long_gap_age_transition_fixed_models.csv) | .csv |
