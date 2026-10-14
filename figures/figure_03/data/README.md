# Figure 3 source data

| File | Purpose |
| --- | --- |
| example_coordinates.npz | Four displayed trajectories and event-marker endpoints, in 15-by-15 maze coordinates. Prefixes A, A2, B and B2 follow session order in example_sessions.csv. |
| example_sessions.csv | Session and neuron identities, observation counts and C_raw-based display thresholds for the example. |
| convergence_curves.csv | Plot-ready group means, uncertainty and support masks for the three moving-window curves. |
| first_window_observations.csv | One first-30-minute correlation per recording cycle and measure, including animal IDs and cycle index; four measures form the BH4 family. |
| successive_cycle_observations.csv | One correlation per registered pair and measure, with animal IDs and experience index; four measures form the BH4 family. |
| convergence_cycle_slopes.csv | Cycle slopes for three measures and four window definitions, supporting the complete BH12 family. |

Map values use S smoothed with Gaussian half-width at half-maximum 1 s.
Map comparisons require 0.066 summed stored S units per retained minute on both
days. Within-cycle ensemble rates use all components; registered ensemble
rates retain finite matched rates and exclude only neurons silent in both
intervals. Activity-marker positions are illustrative display observations.
