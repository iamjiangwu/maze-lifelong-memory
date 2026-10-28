# Source Data for Extended Data Figure 5

`gallery_maps.npz` contains 64 normalized 45 × 45 display maps. `gallery_examples.csv` maps each array to its visual rank, session, neuron, correlation, scale and highlighted status. Panels b–d contain 144 pair observations per metric; e contains 36 context observations. Ensemble-rate and context comparisons retain finite rate vectors, exclude both-zero pairs and do not apply the place/PV activity gate.

Pairs receive equal weight; repeated pairs are clustered by animal. Each metric has six paired day-combination contrasts. Exact animal sign flips and CR1 inference each receive BH6 within that metric. Panel e uses one animal-clustered maze-minus-outside comparison.

`record_type` (or `row_type`) identifies observations, display summaries and inference rows. IDs identify animals, sessions, pairs, neurons or chains. Correlations and policy probabilities are dimensionless; decoder error is in maze tiles. Undefined values are empty cells. Descriptive coordinates are supplied directly; statistical tests are recalculated from observations.
