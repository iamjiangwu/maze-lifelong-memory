# Figure 5: data

Level representations and their association with exploration efficiency. [Column and array index](index.csv).

| Panel | Inputs and observation unit |
|---|---|
| a | Level-code schematic drawn by code. |
| b | `level_map_neuron_vectors.csv`: one registered neuron pair per row with source/target values at levels 0–6. `neuron_display_order.npy` fixes the heatmap order. |
| c | `level_pv_pair_matrices.csv`: one session pair × source level × target level per row; `level_pv_matrix.csv` supplies the displayed mean matrix. |
| d | `level_map_distribution.npz`: neuron-level observed correlations and a saved shuffle histogram with bin edges. |
| e | `level_map_exploration_rows.csv`: one long-gap session pair's level-map correlation and later exploration efficiency; the JSON contains fit results. |

Levels 0–6 index the seven tree depths, not horizontal/vertical spatial partitions. `train_level_*`/`test_level_*` contain source/target level activity values; `level_corr` is their correlation. Level PV correlations instead compare neural population vectors across source/target depths. `supported`, `valid_count`, `n_common_levels` and `included_in_primary` describe the corresponding support and eligibility. Undefined values are excluded, not zero-filled.

`exploration_efficiency` is 32 divided by end-node visits needed to reach 32 distinct end nodes on return. Neuron-level distributions, cell-wise matrix means and pair-level behavioral fits have different averaging units; the panel-e test uses standard errors clustered by animal.

Panel e fits **level-map correlation as response** and exploration efficiency as predictor, using 36 session pairs from 14 animals and a nominal two-sided p value. The related [ED10a](../../extended_data_10/data/README.md) analysis uses 30 pairs on common support across neural readouts, fits exploration efficiency as response, and applies row-wise BH3. Pearson r is symmetric, but the slope test with standard errors clustered by animal depends on this direction and cohort.

In the fit JSON, `corrected_pair*` records the registration-corrected pair `842day1->842-3day1`, its 75 supported neurons and mean level-map correlation. `refreshed_pairs` identifies refreshed input pairs; `legacy_estimator_reproduction` records an upstream consistency check, not an additional statistical test or a check run by this reproduction package.

| File | Format |
|---|---|
| [level_map_distribution.npz](level_map_distribution.npz) | .npz |
| [level_map_exploration_rows.csv](level_map_exploration_rows.csv) | .csv |
| [level_map_exploration_statistics.json](level_map_exploration_statistics.json) | .json |
| [level_map_neuron_vectors.csv](level_map_neuron_vectors.csv) | .csv |
| [level_pv_matrix.csv](level_pv_matrix.csv) | .csv |
| [level_pv_pair_matrices.csv](level_pv_pair_matrices.csv) | .csv |
| [neuron_display_order.npy](neuron_display_order.npy) | .npy |
