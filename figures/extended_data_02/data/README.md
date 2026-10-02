# Extended Data Figure 2: data

First-30-minute junction turning biases and saved biased-walk results. `window_minutes` is 30 throughout this figure's retained analysis; it is time in the maze, not a 20-minute sensitivity window. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a | Turning-bias definitions drawn by code. |
| b–c | `primary_30min_strict_joint_equal_junction_policy_rows.csv` and `primary_30min_strict_joint_policy_cluster_stats.csv`: session probabilities and group contrasts on the joint supported set. |
| d, h | `matched_counterfactual_session_rows.csv` and the `primary_30min_strict_joint_*contribution*`/`*gap_recapitulation*` summaries: matched return/naïve simulations, one-bias contributions and reconstructed efficiency gaps. The shorter `panel_d_*`/`panel_h_*` files supply the corresponding display summaries. |
| e, g | `fixed_time_policy_rows.csv`: natural-support session probabilities; `equal_junction_depth_group_cluster_stats.csv` supplies depth contrasts. |
| f, i | `equal_junction_simulation_session_rows.csv` and `equal_junction_calibration_summary.csv`: measured and simulated efficiency per session for four-bias and depth-varying walks. |

All four biases are conditional probabilities (0–1): `Bf` = P_SF (on arrival from the stem, enter a child rather than reverse); `Ba` = P_SA (conditional on entering a child, choose the alternating branch); `Lf` = P_BF (on arrival from a child, leave that branch rather than reverse); `Lo` = P_BS (conditional on leaving it, take the stem rather than the sibling). `Lo_shallow/mid/deep` split P_BS by depth. Source group codes `naive`, `LTM`, `experienced` mean naïve, return and second day.

Support flags and event counts (`n_Bf`, etc.) define estimability. Use the joint set for the comparisons that require it; do not combine it with the natural-support set. Simulations are saved derived results, not new random walks on every build. Efficiency is distinct end nodes reached divided by the visits needed to reach the target; `delta_*` and `contribution_*` are differences in that ratio. Repeated sessions are linked by `animal_id`; BH4 is applied per four-bias contrast and BH3 per three-depth contrast.

| File | Format |
|---|---|
| [equal_junction_calibration_summary.csv](equal_junction_calibration_summary.csv) | .csv |
| [equal_junction_depth_group_cluster_stats.csv](equal_junction_depth_group_cluster_stats.csv) | .csv |
| [equal_junction_policy_group_cluster_stats.csv](equal_junction_policy_group_cluster_stats.csv) | .csv |
| [equal_junction_simulation_session_rows.csv](equal_junction_simulation_session_rows.csv) | .csv |
| [fixed_time_policy_rows.csv](fixed_time_policy_rows.csv) | .csv |
| [matched_counterfactual_session_rows.csv](matched_counterfactual_session_rows.csv) | .csv |
| [panel_d_policy_contributions.csv](panel_d_policy_contributions.csv) | .csv |
| [panel_h_four_vs_depth_gap.csv](panel_h_four_vs_depth_gap.csv) | .csv |
| [primary_30min_strict_joint_equal_junction_policy_rows.csv](primary_30min_strict_joint_equal_junction_policy_rows.csv) | .csv |
| [primary_30min_strict_joint_gap_recapitulation_summary.csv](primary_30min_strict_joint_gap_recapitulation_summary.csv) | .csv |
| [primary_30min_strict_joint_policy_cluster_stats.csv](primary_30min_strict_joint_policy_cluster_stats.csv) | .csv |
| [primary_30min_strict_joint_policy_contribution_summary.csv](primary_30min_strict_joint_policy_contribution_summary.csv) | .csv |
