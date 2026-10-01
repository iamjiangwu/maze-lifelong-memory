# ED10 caption and source guide

## Suggested caption

**Extended Data Fig. 10 | Shared spatial-code quality covaries with later navigation.**
**a,** Pooled pair-level Pearson correlations between six Day-1-source neural
readouts and three later-session behavioral measures. The original five-row
PowerPoint inventory is retained and whole-maze decoding is added immediately
below reward decoding as a general-decoder-quality control. Both decoder rows
use denoised C calcium activity, the same K = 166 neuron draws and exactly 2,000
target frames per pair; whole-maze decoding is not an all-frame or occupancy-
balanced estimator. All neural readouts use
the current exclusion-clean, no-manual-mask K = 166 fixed-neuron authority
(30 source-to-target session pairs, 14 mice). The three outcomes are
reward-approach speed advantage, elapsed time to the first sustained long bout,
and exploration efficiency; lower first-long-bout time denotes earlier onset.
**b,** Commonality
partition of the pooled reward-proximal-decoder and level-map associations with
first long bout time and exploration efficiency. Explained-variance partitions
are identical for elapsed bout time and its sign-reversed earlier-oriented
form. Bars show reward-decoder-unique,
shared, and level-map-unique explained variance. Here decoder unique means
reward decoder unique relative to the level map, not relative to whole-maze
decoding; these partitions are
associational and do not imply mediation. Technical neuron draws were averaged
within session pair and do not increase biological n.

## Required source-data tables

- `CURRENT_FIXED_NEURON_PAIR_ROWS.csv`: one biological row per session pair.
- `POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv`: the exact 6 x 3 panel-a
  matrix using `log_reward_over_pooled` for the first column and including the
  whole-maze decoder control.
- `COMPACT_CORRELATION_MATRIX.csv`: the separately labeled first-30-min
  sensitivity matrix and adjusted companions.
- `REWARD_LEVEL_COMMONALITY.csv`: panel-b pooled and adjusted partitions.
- `REWARD_LEVEL_PROFILE_TESTS.csv`: pooled, stage, speed, stage-plus-speed, and
  equal-mouse profile tests.
- `SHARED_SPECIFIC_COMPONENT_EFFECTS.csv`: orthogonal common-quality and
  specificity analyses.
- `WITHIN_BETWEEN_MOUSE_DECOMPOSITION.csv`: within- versus between-mouse
  supporting analysis.
- `TECHNICAL_DRAW_PROFILE_STABILITY.csv`: all 50 coordinated K = 166 technical
  draws, explicitly not biological replicates.
- `REWARD_SPEED_EVENT_SUPPORT_SENSITIVITY.csv`: sparse-event and whole-session
  outcome checks.
- `NEURAL_ELIGIBILITY_AUDIT.csv` and `NEURAL_INPUT_IDENTIFIER_SCAN.csv`:
  fail-closed cohort and forbidden-session checks.

## Claim boundary

Use: “Across return stages, stronger Day-1 spatial-map quality covaried with
more efficient later navigation; the relationship was strongest for shared
spatial-code quality and did not uniquely isolate reward-location coding.”

Do not use: “independent replication,” “unique reward code,” “mediation,” or
“stage-independent double dissociation.” Main Fig. 4f and ED10 analyze the same
30 pairs and 14 mice and therefore provide complementary views of one cohort.
