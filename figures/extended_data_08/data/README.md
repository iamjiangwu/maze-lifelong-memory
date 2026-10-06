# Extended Data Figure 8: data

Reward-location and route controls. Current display letters are mapped below because some filenames and `panel` values retain source-analysis letters. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a | `accepted_h.csv`: mean decoding error per spatial bin; `accepted_maze_wall.csv` supplies maze geometry. |
| b | `accepted_b.csv`: animal means of reward/control errors after equalizing frames or visits in source and target sessions; `accepted_stats.csv` supplies tests. |
| c | `accepted_c.csv`: reward/control PV correlations for each registered session pair. |
| d | `signal_comparison_primary_pairs.csv` (`panel=main`) and `replication_day2_pairs.csv`: first-/second-day reward-route enrichment; associated `*_numbers.csv` give summaries/tests. |
| e–g, i | `signal_comparison_primary_pairs.csv`: reliability (`panel=e`), equal-frame (`f`), reward-delivery (`i`, displayed g) and starting-node (`h`, displayed i) controls. `candidate_paired_display_summary.csv` fixes matched-animal display summaries. |
| h | `accepted_routes.csv` and maze walls: the displayed route schematic. |
| j | `strength_checks_profiles.csv` and `strength_checks_numbers.csv`: per-animal/pair/bin direction profiles and their comparisons. |

Decoder errors are Euclidean distance in 15 × 15 spatial bins. Internal `reward_proxy` fields denote reward decoding error. In route data, `port` means the reward-location region, `away` its comparison region, and their difference is reward enrichment. `*_sh` are neuron-ID-shuffled counterparts. `group` distinguishes direction or control route; `bin` is standardized route position. PV/map correlations and reliability ratios are dimensionless.

Route analyses first average pairs within animal; matched paired displays require both values to be finite. In particular, reliability displays use the same 13 animals for both markers in each direction. A missing ratio is undefined, not zero. Panel c retains pair-level observations and inference with standard errors clustered by animal. Saved route tests are exact sign-flip tests on animal-mean effects; see the numerical statistics fields for each correction family. `route_statistics.csv` lists the ten route tests drawn in panels e, f, g, i and j: estimate, 95% interval, exact p, correction family, adjusted p and the drawn mark. `display_panel` gives the displayed letter and `analysis_panel` the source-analysis letter used in the other files.

When a mouse value or its 95% interval reaches the significance bracket in panels e, f or i, the renderer raises the bracket and the upper axis limit together in steps of 5% of the axis range.

| File | Format |
|---|---|
| [accepted_b.csv](accepted_b.csv) | .csv |
| [accepted_c.csv](accepted_c.csv) | .csv |
| [accepted_h.csv](accepted_h.csv) | .csv |
| [accepted_maze_wall.csv](accepted_maze_wall.csv) | .csv |
| [accepted_routes.csv](accepted_routes.csv) | .csv |
| [accepted_stats.csv](accepted_stats.csv) | .csv |
| [candidate_paired_display_summary.csv](candidate_paired_display_summary.csv) | .csv |
| [replication_day2_numbers.csv](replication_day2_numbers.csv) | .csv |
| [replication_day2_pairs.csv](replication_day2_pairs.csv) | .csv |
| [route_statistics.csv](route_statistics.csv) | .csv |
| [signal_comparison_primary_numbers.csv](signal_comparison_primary_numbers.csv) | .csv |
| [signal_comparison_primary_pairs.csv](signal_comparison_primary_pairs.csv) | .csv |
| [strength_checks_numbers.csv](strength_checks_numbers.csv) | .csv |
| [strength_checks_profiles.csv](strength_checks_profiles.csv) | .csv |
