# Extended Data Figure 3: data

Reward-path dwell and first-visit route choices. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–d | `fluency_plot_rows.csv` and `fluency_statistics.csv`: session/claim means for the specified reward epoch, path and comparison, with corresponding contrasts. |
| e | `route_definition.csv`: the six decision nodes, correct child and required turn along the reward route. |
| f | `depth_small_multiples.csv`: mean and s.e.m. of correct first choices at each decision depth. `session_route.csv` and `route_stats.csv` supply complete six-choice session scores and their associated comparisons. |

`mean_decision_time_s` is junction dwell time in seconds; `n_events` records the contributing visits. `claim_id`, `reward_epoch`, `path_class` and `included_for_displayed_primary` separate the displayed comparisons. `paper_depth` is the displayed decision index; the maze's internal `tree_depth` and node IDs are also given in the route definition.

`route_fraction_correct` is the fraction of the six first decisions scored as reward-directed. The observed/inferred/resolved counts document whether all six choices can be scored. Unsupported choices are not failures. Statistical contrasts keep repeated sessions together by animal; means at each decision depth and whole-route scores are different summaries.

| File | Format |
|---|---|
| [depth_small_multiples.csv](depth_small_multiples.csv) | .csv |
| [fluency_plot_rows.csv](fluency_plot_rows.csv) | .csv |
| [fluency_statistics.csv](fluency_statistics.csv) | .csv |
| [route_definition.csv](route_definition.csv) | .csv |
| [route_stats.csv](route_stats.csv) | .csv |
| [session_route.csv](session_route.csv) | .csv |
