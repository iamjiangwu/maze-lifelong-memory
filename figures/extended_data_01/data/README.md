# Extended Data Figure 1: data

Reward and exploration measurements. The mappings below use the current a–j panel letters; some filenames retain earlier letters. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–b | `bc_group_specific_80_session_curves.csv.gz`: cumulative rewards/long bouts for each session at each time point. |
| c–d | `de_group_specific_80_group_curves.csv`: group means and s.e.m. against maze tile transitions. |
| e | `endpoint_axis_session_rows.csv` and `endpoint_axis_cluster_stats.csv`: one session × endpoint outcome, in maze time and tile transitions, and the group contrasts. |
| f | `tile_time_loglog_rows.csv` and its statistics file: elapsed maze time and tile transitions to the first long bout. |
| g | `panel_h_first30_session_rows_cap60.csv` and its summary: first-30-min speed per session. |
| h | `panel_i_end_node_sequence.csv` and `panel_i_cycle_summary.csv`: displayed coverage-bout examples. |
| i–j | `j_*_time_session_bins.csv`, `k_*_tile_session_bins.csv`: efficiency per session and time/tile bin. |

`time_minutes` is time in the maze; `maze_time_s` is seconds; tile values count transitions, not physical distance. `mode_time_distance_rate_cm_s` is speed in cm/s from retained continuous body-position pairs with the saved 60 cm/s segment cap. A coverage bout ends after reaching 32 distinct end nodes; efficiency is `32 / steps_in_cycle`, where steps count end-node visits. These coverage-bout indices are not recording cycles.

`endpoint_reached`, `display_coverage_ok` and support fractions identify valid measurements/display ranges. `animal_id` groups repeated sessions for inference. The endpoint tests form BH18 over the displayed comparisons and axes; saved `load_ok` or missing values must not be interpreted as zero outcomes.

| File | Format |
|---|---|
| [bc_group_specific_80_session_curves.csv.gz](bc_group_specific_80_session_curves.csv.gz) | .csv.gz |
| [de_group_specific_80_group_curves.csv](de_group_specific_80_group_curves.csv) | .csv |
| [endpoint_axis_cluster_stats.csv](endpoint_axis_cluster_stats.csv) | .csv |
| [endpoint_axis_session_rows.csv](endpoint_axis_session_rows.csv) | .csv |
| [j_group_specific_80_time_session_bins.csv](j_group_specific_80_time_session_bins.csv) | .csv |
| [k_group_specific_80_tile_session_bins.csv](k_group_specific_80_tile_session_bins.csv) | .csv |
| [panel_h_first30_group_summary_cap60.csv](panel_h_first30_group_summary_cap60.csv) | .csv |
| [panel_h_first30_session_rows_cap60.csv](panel_h_first30_session_rows_cap60.csv) | .csv |
| [panel_i_cycle_summary.csv](panel_i_cycle_summary.csv) | .csv |
| [panel_i_end_node_sequence.csv](panel_i_end_node_sequence.csv) | .csv |
| [tile_time_loglog_rows.csv](tile_time_loglog_rows.csv) | .csv |
| [tile_time_loglog_statistics.csv](tile_time_loglog_statistics.csv) | .csv |
