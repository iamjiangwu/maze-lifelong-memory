# Figure 4: data

Position-decoder results, route correlations and the behavioral association. [Column and array index](index.csv).

| Panel | Inputs and observation unit |
|---|---|
| a | `spatial_decoder_example.npz` and its JSON metadata: five example place maps, target activity and a saved position posterior. |
| b | `figure4b_neuron_id_shuffle_pair_summary.csv`: observed and neuron-ID-shuffled mean errors for each source/target session pair; the JSON contains the paired inference. |
| c | `panel_c_example_map_rows.csv`: mean decoding error and frame count in each target spatial bin. |
| d | `panel_d_native_reference_rows.csv`: reward/control decoding errors per eligible session pair and source-day variant; `day_source_statistics.csv` contains the comparisons. |
| e | `panel_e_heatmaps.npz`, `panel_e_pair_profiles.csv`, `panel_e_animal_profiles.csv`, `panel_e_curve.csv` and `panel_e_statistics.csv`: neuron heatmaps, pair/bin and animal/bin route PV profiles, cohort curves and route tests. |
| f | `panel_f_matched_pair_means.csv`: reward decoding error and first long bout time for each matched neural-behavior pair. |

Decoding error is Euclidean distance in 15 × 15 spatial bins. First long bout time is cumulative maze time in seconds to the qualifying path's onset, as defined in the [Figure 1 data guide](../../figure_01/data/README.md). Reward maze node 94, the physical water port and decoder bin 103 are distinct identifiers. Internal `reward_proxy`/`d1_reward_proxy_error` fields are the plotted **reward decoding error**, not an additional proxy outcome. Control bins are 91, 121 and 133.

`group` distinguishes home-to-reward and reward-to-home routes; `bin` uses the same spatial orientation for both directions: 0 at home and 21 at the reward end. The departure heatmap reverses this order for display. `pv` and `pv_sh` are observed and neuron-ID-shuffled correlations. Route profiles average pairs within animal before across-animal summaries. For each animal, reward enrichment is the correlation at bin 21 minus the mean correlation over bins 0–20; the two direction tests compare these animal effects with zero. Pair eligibility and reward/control support flags must be preserved. Panel f uses the ED10a fit definition, with an uncorrected p value rather than ED10a's row-wise BH3.

| File | Format |
|---|---|
| [day_source_statistics.csv](day_source_statistics.csv) | .csv |
| [figure4b_neuron_id_shuffle_pair_summary.csv](figure4b_neuron_id_shuffle_pair_summary.csv) | .csv |
| [figure4b_neuron_id_shuffle_statistics.json](figure4b_neuron_id_shuffle_statistics.json) | .json |
| [panel_c_example_map_rows.csv](panel_c_example_map_rows.csv) | .csv |
| [panel_d_native_reference_rows.csv](panel_d_native_reference_rows.csv) | .csv |
| [panel_e_animal_profiles.csv](panel_e_animal_profiles.csv) | .csv |
| [panel_e_curve.csv](panel_e_curve.csv) | .csv |
| [panel_e_heatmaps.npz](panel_e_heatmaps.npz) | .npz |
| [panel_e_pair_profiles.csv](panel_e_pair_profiles.csv) | .csv |
| [panel_e_statistics.csv](panel_e_statistics.csv) | .csv |
| [panel_f_matched_pair_means.csv](panel_f_matched_pair_means.csv) | .csv |
| [spatial_decoder_example.json](spatial_decoder_example.json) | .json |
| [spatial_decoder_example.npz](spatial_decoder_example.npz) | .npz |
