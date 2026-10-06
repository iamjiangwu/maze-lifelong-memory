# Figure 1: data

Behavioral examples and session-level outcomes. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–d | `maze_3d_original.png` and `panel_a_decision_points.csv` define the illustration/reward route; `trajectory_examples.csv.gz` supplies example trajectory samples. The cycle/timeline schematics are drawn by code. |
| e, h | `endnode_timelines.csv`: visits to maze end nodes in the displayed sessions. |
| f | `reward_acquisition_day1_month_*`: first long bout time for each session, group summaries and contrasts across return months; `return_experience_statistics.csv` records the within-return trend. |
| g | `exploration_first_chunk.csv`: one session's first complete half-maze coverage bout, and a final contrast row. |
| i | `long_bout_group_*`: session values, summaries and contrasts across reward contexts. |
| j | `session_route.csv`, `route_group_stats.csv`, `route_stats.csv`: first-visit route scores, summaries and contrasts. |

`lb0` is cumulative time in the maze, in seconds, to the onset of the first long bout. A long bout is an uninterrupted path ending at the water port with more than six consecutive transitions between adjacent tree nodes, each reducing distance to the port. The timestamp is the path's onset, not its arrival at the port.

`first_chunk_efficiency` is 32 divided by the number of end-node visits needed to reach 32 distinct end nodes during exploration; a new visit requires leaving and returning to an end node. The last row of `exploration_first_chunk.csv` has `record_type` = `contrast`. It gives the return minus naive difference in first-chunk efficiency over the 79 sessions from 29 animals in the table, from an ordinary least-squares regression on a return indicator with standard errors clustered by animal (CR1) and a two-sided t test with G−1 degrees of freedom. Session rows leave the contrast columns empty, and the contrast row leaves the session columns empty. `route_fraction_correct` is the fraction of correct first choices across the six reward-path decision nodes (0–1); `all6_resolved` and the decision-count fields describe support. `animal_id`/`animal` links repeated sessions from one animal. Missing or unsupported outcomes are not zeros. Group summaries and inferential contrasts have separate units; see [statistical families](../../../docs/STATISTICS.md).

| File | Format |
|---|---|
| [endnode_timelines.csv](endnode_timelines.csv) | .csv |
| [exploration_first_chunk.csv](exploration_first_chunk.csv) | .csv |
| [long_bout_group_clustered_contrasts.csv](long_bout_group_clustered_contrasts.csv) | .csv |
| [long_bout_group_points.csv](long_bout_group_points.csv) | .csv |
| [long_bout_group_summary.csv](long_bout_group_summary.csv) | .csv |
| [maze_3d_original.png](maze_3d_original.png) | .png |
| [panel_a_decision_points.csv](panel_a_decision_points.csv) | .csv |
| [return_experience_statistics.csv](return_experience_statistics.csv) | .csv |
| [reward_acquisition_day1_month_clustered_contrasts.csv](reward_acquisition_day1_month_clustered_contrasts.csv) | .csv |
| [reward_acquisition_day1_month_points.csv](reward_acquisition_day1_month_points.csv) | .csv |
| [reward_acquisition_day1_month_summary.csv](reward_acquisition_day1_month_summary.csv) | .csv |
| [route_group_stats.csv](route_group_stats.csv) | .csv |
| [route_stats.csv](route_stats.csv) | .csv |
| [session_route.csv](session_route.csv) | .csv |
| [trajectory_examples.csv.gz](trajectory_examples.csv.gz) | .csv.gz |
