# ED10 current fixed-neuron synthesis: pooled profile and specificity controls

## Bottom line

The corrected fixed-neuron analysis is not a weaker version of the old result. It asks several different questions that had been mixed together in the previous display. When the PowerPoint's original pooled comparisons are restored, the main neural-behavior pattern remains strong. The stricter controls then show what the pattern can—and cannot—mean.

Across 30 session pairs from 14 mice, better preservation of hippocampal spatial representations accompanied better later behavior. The clearest interpretation is a **shared spatial-code quality signal**: reward-proximal decoding and broader spatial maps tend to be good in the same recordings, and that common quality covaries with later performance. The data do not establish a unique reward code or a stage-independent double dissociation.

## What remained strong

The pooled reward-versus-level-map behavioral profile is robust. Reward-proximal decoding was more related to earlier onset of a sustained route, whereas level-map stability was more related to exploration efficiency. The crossover was 0.406 (CR1 P=0.002361; exact P=0.008545; maxT3 P=0.01831). Leaving out each mouse in turn kept the crossover positive in 14/14 analyses (0.339 to 0.462). Equal weighting of mice also preserved it (0.465; exact maxT3 P=0.04358).

The result is not an accident of one neuron subsample. Across the 50 coordinated K=166 technical draws, the crossover was positive in 50/50 draws (mean 0.391, range 0.186–0.541); the expected unique-variance ordering appeared in 46/50 draws. These draws measure technical selection stability, not biological replication or a confidence interval.

The pooled variance partition is similarly intuitive:

- Earlier sustained routing: 12.2% reward-decoder unique, 8.8% shared, and 2.4% level-map unique variance.
- Exploration efficiency: 2.2% reward-decoder unique, 17.3% shared, and 35.1% level-map unique variance.

These percentages describe overlapping associations; they are not mediation fractions or causal pathways.

## The important new result: common quality is more convincing than specificity

Because the neural readouts are highly correlated, we rotated each pair into two orthogonal parts: a common-quality component and a specificity component. This avoids making a shared signal disappear merely because two correlated predictors compete in one regression.

For earlier sustained routing, the common decoder-quality component remained positive after stage and generic-speed adjustment (beta=0.376, CR1 P=0.0163; maxT2 P=0.08301), whereas reward-location specificity did not (beta=-0.117, P=0.6172).
For exploration, the common place/level-map stability component was strongly positive (beta=0.489, CR1 P=0.001705; maxT2 P=0.04504), whereas level-map specificity was not (beta=-0.060, P=0.575).

The reward-bin decoder also did not outperform the whole-maze decoder for the first-long-bout outcome: their pooled correlations were r=0.458 and r=0.479, respectively. In the strict joint model, reward location added essentially no unique evidence beyond whole-maze decoding, stage, and generic speed (beta=-0.001, P=0.9989). The neural signal should therefore be described as reward-proximal or general spatial decodability, not a uniquely reward-specific code.

## Does the association reflect stable mouse differences or changes within a mouse?

The decoder/route relation was mainly between mice: the between-mouse component was positive (beta=0.398, P=0.01705), but the within-mouse component was not (beta=-0.142, P=0.3593). The equal-mouse correlation was r=0.601 (P=0.0231). This is consistent with a stable recording/animal quality axis and does not establish that improving decoding within one mouse causes earlier routing.

The spatial-map/exploration relation was broader: both the between-mouse component (beta=0.460, P=0.001063) and the within-mouse component (beta=0.211, P=0.02411) were positive, and the equal-mouse correlation was r=0.693 (P=0.005987). This is the most compelling evidence that preservation of spatial-map organization accompanies efficient later exploration.

## Why the previous revision looked substantially worse

Three display/estimand changes—not a data collapse—caused the visual difference:

1. The PowerPoint showed pooled Pearson correlations, whereas the recent synthesis showed stage- and speed-adjusted conditional coefficients.
2. The recent figure removed the population-vector row, reversed the first two behavior columns, and replaced the commonality/policy panels with a large sensitivity forest.
3. The reward/control-speed endpoint changed from a whole-session measure to a first-30-min matched-terminal measure. Those two behavior measures correlate only r=0.538.

The current neural-source correction itself is tiny: across unchanged behavioral endpoints, the maximum old-to-current pooled-correlation shift attributable to the neural source was 0.1308. The no-manual-mask change alters the pooled crossover by only about 0.008. Thus the scientific source correction did not erase the original pattern; changing the displayed estimand did.

## Where stage and movement matter

The pooled crossover remained positive after generic-speed adjustment (0.538; maxT3 P=0.0437) but was weaker after stage adjustment (0.276; maxT3 P=0.4271) and after stage plus speed (0.315; maxT3 P=0.5153). The exciting pooled profile is therefore **stage-dependent**, not a universal stage-independent double dissociation.

The first-30-min reward/control speed outcome also has sparse event support in three rows. The place-beyond-level estimate stayed similar when requiring at least two events in each arm (beta=0.780, n=28) or at least three (beta=0.787, n=27), compared with all rows (beta=0.759, n=30), but uncertainty remained substantial. Using the whole-session behavior measure produced a much smaller estimate (beta=0.543). This sensitivity belongs in Source Data, not as another panel.

## How Main Figure 4f fits

Main Figure 4f and ED10 use the same 30 pairs and 14 mice, so they are complementary analyses of one cohort, not independent replications. Main Figure 4f remains a clear pooled observation: better Day-1-source spatial decoding accompanies earlier sustained navigation after the gap. ED10 supplies the missing interpretation controls. It shows that the association is not uniquely reward-location specific and is partly structured by return stage and general movement.

A concise claim that fits both figures is: **Across return stages, stronger Day-1 spatial-map quality covaried with more efficient later navigation; the relationship was strongest for shared spatial-code quality and did not uniquely isolate reward-location coding.**

## What the revised figure shows

Panel a retains the PowerPoint's original column order and five readouts, then adds whole-maze decoding immediately below reward decoding as a general-decoder-quality control. The resulting 6 x 3 matrix uses reward-approach speed advantage, first long bout time, and exploration efficiency. The displayed correlations use elapsed first-long-bout time directly, so better neural stability is expected to correlate negatively with that column. These are descriptive effect sizes; the panel deliberately contains no stars or cellwise P values. The first-30-min matched-terminal speed endpoint remains a separately labeled Source Data sensitivity.

Panel b restores the pooled reward-versus-level-map commonality bars for first long bout time and exploration. Explained-variance partitions are invariant to multiplying bout time by -1, so the bars are unchanged from the earlier-oriented Source Data calculation. Its Reward-decoder unique segment means unique relative to the level map, not unique relative to whole-maze decoding. The strict conditional models, shared-versus-specific rotations, stage/speed controls, equal-mouse analysis, within/between decomposition, event-support sensitivity, and 50-draw stability are kept in the report and Source Data rather than crowded into the figure.

The separate clean36 depth-weighted policy bridge remains useful as a distinct supporting family because it asks a different question in a different cohort. It should not be called an independent replication of the 30-pair matrix.

## Data integrity and eligibility

All displayed neural results use the current no-manual-mask K=166 authority: 30 pairs, 14 mice, the same selected neurons in decoder and raw-map calculations, 85 matched spatial bins, and 125 raw-C rows per bin/session. The live neural-eligibility contract is applied before technical QC in selection, decoder, raw-map and synthesis steps. Technical draws are averaged within pair before inference.

Animals 840 and 841 are behavior-only and contribute no neural data. The bad-quality 488 nine-month session, the sick 579 six-month session, and every other locked neural exclusion are absent. The six-month Day1/Day2 session pairs for animals 1636, 1637, 1639, 325 and 347 are absent because they belong to a different experiment. Earlier valid 488 and 579 intervals remain only when they do not touch the excluded sessions.

## Boundaries on interpretation

- Associations do not show that one neural code causes behavior.
- The first long bout combines route formation, exploration, and locomotor state.
- Reward/control speed is measured after reward discovery and is not pure initial recall.
- Fixed K equalizes ensemble size, not registration quality, occupancy, or all recording-state differences.
- Pair-weighted estimates give mice with multiple return intervals more rows; equal-mouse and LOMO results are therefore reported explicitly.
- The 50 draws are technical stability checks, never biological n.

## Source-data guide

- `POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv`: the exact displayed 6 x 3 pooled matrix using the PowerPoint's whole-session speed endpoint and the added whole-maze decoder control.
- `COMPACT_CORRELATION_MATRIX.csv`: the first-30-min pooled and adjusted sensitivity matrix.
- `REWARD_LEVEL_COMMONALITY.csv`: pooled and adjusted variance partitions.
- `REWARD_LEVEL_PROFILE_TESTS.csv`: pooled, stage, speed, stage+speed, and equal-mouse crossover tests.
- `SHARED_SPECIFIC_COMPONENT_EFFECTS.csv`: common-quality versus specificity components.
- `WITHIN_BETWEEN_MOUSE_DECOMPOSITION.csv`: stable mouse differences versus within-mouse variation.
- `TECHNICAL_DRAW_PROFILE_STABILITY.csv`: all 50 coordinated K=166 draw results.
- `REWARD_SPEED_EVENT_SUPPORT_SENSITIVITY.csv`: first-30-min and whole-session behavior support checks.
- `ESTIMAND_RECONCILIATION.csv`: old pooled, current-source pooled, current endpoint, and adjusted estimates side by side.
- `NEURAL_ELIGIBILITY_AUDIT.csv` and `NEURAL_INPUT_IDENTIFIER_SCAN.csv`: fail-closed proof that behavior-only animals and excluded neural sessions are absent.
