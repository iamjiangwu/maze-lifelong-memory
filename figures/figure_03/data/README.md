# Figure 3: data

Day-2-reference curves and the observations used in experience-trend fits. [Column and array index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a | The neuron-example NPZ supplies one registered neuron across four sessions: A = Day i, A2 = Day i+1, B = Day i+90, B2 = Day i+91. |
| b–d | Analysis schematics drawn by code; no biological measurements are required. |
| e, h, k (first column) | `day2_template_group_curve_summary.csv` supplies mean, s.e.m. and animal count at each curve position; `day2_template_curve_support.csv` supplies one support threshold per metric/experience curve. Individual moving-window curves are not included. |
| f, i, l (second column) | `combined_first30_points_all_metrics.csv`: one animal × experience × metric observation for alignment of the first 30 minutes with that cycle's full Day-2 representation. The associated statistics file records the fits. |
| g, j, m (third column) | `long_gap_experience_points.csv`: one session-pair × metric correlation across successive recording cycles; `long_gap_experience_statistics.csv` records the fits. |

Metrics are dimensionless correlations. `experience_index`/`experience` identifies the recording cycle; it is not calendar age. For time curves, `x_value` is cumulative maze time at the end of each moving window, in seconds (`x_unit=sec`), and is displayed in minutes; `window_size=30` is minutes. `mean` and `sem` summarize pairs within animal in each source cohort, then pool cohort moments. For each curve, the renderer compares its own `n_animals` values with `half_max_threshold` and applies the displayed time limits; it does not join a pointwise mask by time. NaN support fields can be inapplicable to a metric, not failed measurements.

The displayed curve and band independently smooth `mean` and `sem` with a centred five-point moving average (partial windows at edges; missing s.e.m. filled with zero). The band is smoothed mean ± smoothed s.e.m., not a recalculated s.e.m. of smoothed animal trajectories. The second- and third-column shaded bands are ordinary least-squares 95% intervals with n−2 degrees of freedom, treating rows as independent; their stars separately use tests with standard errors clustered by animal and BH4.

The point/statistics files also contain the level-map measure shown in ED9n/o, which belongs to the same BH4 families. Second-column animal observations and third-column pair observations are both needed for their tests with standard errors clustered by animal. The first-column curves only require their plotted summaries and support.

The successive-cycle points in g/j/m include mouse 1616's 14-day interval. Thus the file's `long_gap` name does not mean that every included interval is approximately 90 days; this cohort differs from the 36-pair long-gap cohort in Figure 2.

In the example NPZ, `*_frame` indexes the corresponding full `*_Craw`/`*_S` signal; retained position samples must not be matched by array row alone. `dt` is seconds per frame, `*_x`/`*_y` are nose coordinates and `*_mass_x`/`*_mass_y` are body-position coordinates, normalized to approximately 0–1 across the maze. The renderer uses inferred `S` activity and noise estimated from `Craw` to draw events on the trajectories.

| File | Format |
|---|---|
| [451-3day1_to_451-6day1__row0161_a190_b267.npz](451-3day1_to_451-6day1__row0161_a190_b267.npz) | .npz |
| [combined_first30_points_all_metrics.csv](combined_first30_points_all_metrics.csv) | .csv |
| [combined_first30_stats_all_metrics.csv](combined_first30_stats_all_metrics.csv) | .csv |
| [day2_template_curve_support.csv](day2_template_curve_support.csv) | .csv |
| [day2_template_group_curve_summary.csv](day2_template_group_curve_summary.csv) | .csv |
| [long_gap_experience_points.csv](long_gap_experience_points.csv) | .csv |
| [long_gap_experience_statistics.csv](long_gap_experience_statistics.csv) | .csv |
