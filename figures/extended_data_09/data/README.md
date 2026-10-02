# Extended Data Figure 9: data

Level-code persistence, controls and experience trends. The mappings use the current 15-panel a–o layout. [Column and array index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–c | `plot_data.npz` contains level-PV observations and shuffle histogram; `null_draws.csv` contains cohort-mean PV null draws; `mean_null.npz` contains the level-map mean null. `pair_summary.csv` and `summary.json` describe their support and weighting. |
| d | `decoder_points.csv`: observed/shuffled decoding accuracy per session pair; `decoder_summary.csv` and `decoder_stats.json` contain summaries and the paired test. |
| e–f | `transition.csv`, `shift.csv`: preferred-level transition/shift distributions. `preferred_level_pair_observations.csv` and its statistics file supply pair-level inputs and tests. |
| g–h | `maze_cells.csv`/`maze_walls.csv`: partition schematic; `pair_values.csv`, `paired_contrasts.csv`, `statistics.csv`: horizontal/vertical eight-bin map correlations, seven-level map correlations and their contrasts per session pair. |
| i | `joint_cloud.csv`: one neuron pair's place-map and level-map correlation. |
| j | `within_level_pairs.csv` retains the 36 pair-level observed/null correlations used by the paired plot and test. `within_level_nulls.csv` supplies 1,000 cohort-mean null draws for the histogram; `within_level_stats.csv` records the test. |
| k–l | `gap.csv`/`level_gap_pair_values.csv`: means by Days apart and their pair inputs; `consecutive_points.csv`/`consecutive_stats.csv`: preceding-versus-following gap observations and fits. |
| m | `learning.csv`: mean, s.e.m., animal count and support mask for within-session level-reference curves. |
| n–o | `first30_points.csv`/`first30_stats.csv`: animal-by-cycle early alignment; `cycles.csv`/`cycles_stats.csv`: long-gap pair correlations by cycle. |

Correlations and decoding accuracy (fraction correct) are dimensionless. Levels are tree-depth categories; a shift distance is a number of levels, not physical distance. Preferred-level transitions exclude the diagonal where marked. Shift curves pool neurons that changed preference, whereas their tests equally weight 36 session-pair observations from 14 animals, with standard errors clustered by animal and BH2 across two one-sided tests.

`continuous_cloud_included`, support counts and explicit masks determine eligibility. Null draws are saved technical repetitions, not independent animals. In `within_level_nulls.csv`, `repeat` identifies a shuffle repetition, `mean_correlation` equally averages its 36 pair-level correlations, and `n_pairs` records that support. Gap means equally weight session pairs and use descriptive 95% Student's t intervals. Panels l, n and o each share their four-measure correction family with the corresponding Figure 2 or Figure 3 panels; they are not extra unshown tests. Curve times use the stated seconds/minutes fields, and cycle indices are not calendar age.

| File | Format |
|---|---|
| [consecutive_points.csv](consecutive_points.csv) | .csv |
| [consecutive_stats.csv](consecutive_stats.csv) | .csv |
| [cycles.csv](cycles.csv) | .csv |
| [cycles_stats.csv](cycles_stats.csv) | .csv |
| [decoder_points.csv](decoder_points.csv) | .csv |
| [decoder_stats.json](decoder_stats.json) | .json |
| [decoder_summary.csv](decoder_summary.csv) | .csv |
| [first30_points.csv](first30_points.csv) | .csv |
| [first30_stats.csv](first30_stats.csv) | .csv |
| [gap.csv](gap.csv) | .csv |
| [joint_cloud.csv](joint_cloud.csv) | .csv |
| [learning.csv](learning.csv) | .csv |
| [level_gap_pair_values.csv](level_gap_pair_values.csv) | .csv |
| [maze_cells.csv](maze_cells.csv) | .csv |
| [maze_walls.csv](maze_walls.csv) | .csv |
| [mean_null.npz](mean_null.npz) | .npz |
| [null_draws.csv](null_draws.csv) | .csv |
| [pair_summary.csv](pair_summary.csv) | .csv |
| [pair_values.csv](pair_values.csv) | .csv |
| [paired_contrasts.csv](paired_contrasts.csv) | .csv |
| [plot_data.npz](plot_data.npz) | .npz |
| [preferred_level_pair_observations.csv](preferred_level_pair_observations.csv) | .csv |
| [preferred_level_statistics.csv](preferred_level_statistics.csv) | .csv |
| [shift.csv](shift.csv) | .csv |
| [statistics.csv](statistics.csv) | .csv |
| [summary.json](summary.json) | .json |
| [transition.csv](transition.csv) | .csv |
| [within_level_nulls.csv](within_level_nulls.csv) | .csv |
| [within_level_pairs.csv](within_level_pairs.csv) | .csv |
| [within_level_stats.csv](within_level_stats.csv) | .csv |
