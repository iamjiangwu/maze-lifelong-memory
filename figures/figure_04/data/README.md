# Figure 4 source data

| File | Purpose |
| --- | --- |
| spatial_decoder_example.npz and spatial_decoder_example.json | Five illustrative maps and traces, one posterior, neuron identities and displayed coordinates for panel a. |
| whole_maze_pairs.csv | Observed and mean neuron-identity-shuffled decoding errors for 36 registered pairs from 14 animals. |
| example_error_map.csv | Mean error and sample count per occupied maze tile for the mouse 579 example. |
| reward_control_pairs.csv | Reward and three control-bin errors for 30 rewarded-source pairs, with animal IDs. |
| route_heatmaps.npz | Four peak-normalized display matrices in the source-session peak order; normalization affects display only. |
| route_pair_profiles.csv | Observed and mean shuffled standardized route PV profiles for each pair and route bin. |
| decoder_behavior_pairs.csv | Matched 181-neuron decoder means and later first-long-bout onset time for 30 pairs from 14 animals. |

Decoder errors are Euclidean distances in maze tiles. The target is grid bin
103, adjacent to physical reward node 94. Route bins are 3.8 cm wide over the
83.6 cm route; profiles are oriented with reward in the last bin. Each mouse
receives equal weight after its pairs are averaged. Panel f fits behavior on
decoding error and uses an animal-clustered nominal test and a 5,000-draw
whole-animal bootstrap band.
