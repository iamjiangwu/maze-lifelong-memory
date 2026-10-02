# Extended Data Figure 7: data

Age associations conditioned on recording experience. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–c | `first30_plot_ready.csv`: one animal/recording-cycle × metric's first-30-min alignment to its Day-2 reference, with age and cycle index. `first30_age_models.csv` contains fitted age coefficients. |
| d–f | `long_gap_age_joined.csv`: one long-gap session pair × metric with source/target ages and transition. `long_gap_age_transition_fixed_models.csv` contains the transition-adjusted fits. |

`age_day1_days`, `age_src_days` and `age_dst_days` are calendar age in days; `experience_index` and `transition` are recording-experience categories, not age. The upper panels use Day-1 age, the lower panels destination-session age. `value` is a dimensionless neural correlation. `beta_per_30d` is the correlation change per 30 days of age after controlling for experience/transition.

The fitted models weight the included observations equally and use standard errors clustered by animal; they do not first average all of an animal's pairs into one point. `age_control_included`, exclusions and model specification fields record eligibility. The lower family includes the level-map association alongside the three displayed main neural measures (BH4); the upper displayed family is BH3.

| File | Format |
|---|---|
| [first30_age_models.csv](first30_age_models.csv) | .csv |
| [first30_plot_ready.csv](first30_plot_ready.csv) | .csv |
| [long_gap_age_joined.csv](long_gap_age_joined.csv) | .csv |
| [long_gap_age_transition_fixed_models.csv](long_gap_age_transition_fixed_models.csv) | .csv |
