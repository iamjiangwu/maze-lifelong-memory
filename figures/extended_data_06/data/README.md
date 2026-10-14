# Extended Data Figure 6: data

Fixed Day-2-reference and sliding-pivot correlations. [Column and format index](index.csv).

Maps and population vectors use denoised calcium `C`, with neuron eligibility determined from deconvolved activity `S`. Fixed reference comparisons in c/e retain neurons whose summed `S` reaches 0.1 per retained minute on both days. Sliding pivot comparisons in d/f apply that threshold separately within each day, so the retained neurons may differ between days. Ensemble rate panels g/h use every neuron, with no `S` threshold. These choices are already reflected in the supplied values.

| Panels | Inputs and observation unit |
|---|---|
| a–b | Fixed-reference and sliding-pivot schematics drawn by code. |
| c, e, g | `day2_template_curve_summary.csv` and `day2_template_curve_support.csv`: cohort mean, s.e.m. and support by curve position, metric and exposure definition. `day2_template_pair_slopes.csv` supplies recording-pair observations for the fixed-reference tests; `day2_template_clustered_statistics.csv` contains the results. |
| d, f, h | `pivot_experience_group_curve_summary.csv` and `pivot_experience_display_limits.csv`: Day-1/Day-2 local-correlation curves and their supported display ranges. `pivot_available_curve_pair_summaries.csv` supplies one recording cycle × metric's slopes and within-curve standard deviations; its statistics CSV contains the tests. |

Correlations are dimensionless. The fixed-reference curve tables contain the windows shown here: 10 or 20 minutes and 3,000 tile transitions. Figure 3 supplies the 30-minute curves; the slope observations and statistics here retain all four definitions for BH12. `x_plot_unit` identifies minutes or thousands of tile transitions (the latter is multiplied by 1,000 for display). `slope_per_hour_or_ktile` is correlation per hour or per 1,000 transitions. Support masks require the recorded minimum number of animals at each point.

Sliding-pivot curve and display-limit tables contain only the displayed 30-minute window. Statistics use the jointly available curve within each recording cycle. `day_i1_minus_day_i_sd` is Day-2 minus Day-1 within-curve standard deviation, not a difference in fitted slopes. Slopes are correlation per hour. Three separate BH3 families cover Day-1 slopes, Day-2 slopes and the variability differences; repeated cycles remain grouped by animal.

| File | Format |
|---|---|
| [day2_template_clustered_statistics.csv](day2_template_clustered_statistics.csv) | .csv |
| [day2_template_curve_summary.csv](day2_template_curve_summary.csv) | .csv |
| [day2_template_curve_support.csv](day2_template_curve_support.csv) | .csv |
| [day2_template_pair_slopes.csv](day2_template_pair_slopes.csv) | .csv |
| [pivot_available_curve_clustered_statistics.csv](pivot_available_curve_clustered_statistics.csv) | .csv |
| [pivot_available_curve_pair_summaries.csv](pivot_available_curve_pair_summaries.csv) | .csv |
| [pivot_experience_display_limits.csv](pivot_experience_display_limits.csv) | .csv |
| [pivot_experience_group_curve_summary.csv](pivot_experience_group_curve_summary.csv) | .csv |
