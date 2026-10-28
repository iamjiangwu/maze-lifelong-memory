# Statistical analyses

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The command recalculates statistical tests from the included observations and saved null or simulation draws. It writes the resulting tables into one folder per figure and records completion in `BUILD_REPORT.json`. That report confirms execution, not an independent validation of the scientific results.

## Units and uncertainty

Unless specified otherwise, regressions use **standard errors clustered by animal**, with a finite-animal adjustment and a Student's t reference with `G − 1` degrees of freedom. The implementation calls this adjustment CR1. Pearson correlations, fitted slopes, confidence intervals and p values are distinct quantities; an ordinary Pearson p value is not the clustered regression p value.

Figure 2 and ED9 gap curves equally weight session-pair means and show descriptive 95% Student's t intervals. Figure 3's Day-2-reference summaries first average observations within each animal, experience group and time point, then calculate the mean and s.e.m. across animals. For display, Figure 3e/h/k and ED9m separately apply a centred five-point moving average to the mean and s.e.m.; bands show smoothed mean ± smoothed s.e.m. Figure 3 smooths before applying its support/time masks and requires `n_animals >= max(3, ceil(max_n / 2))`. ED9m filters `passes_support_mask` before smoothing, then draws points with at least three animals. These are display transformations, not recalculated s.e.m. values for smoothed individual trajectories.

Figure 3f/i/l and g/j/m use ordinary least-squares 95% confidence bands for the fitted mean, with n−2 degrees of freedom and rows treated as independent. These bands are descriptive: the stars instead use slope tests with standard errors clustered by animal and BH4. ED9n/o's fitted-mean bands instead use the animal-level sandwich covariance with the finite-sample adjustment and t(G−1) reference. Figures 4f and 5e fit later behavior as the outcome and a neural measure as the predictor. Both show ordinary least-squares fits with 95% pointwise confidence bands from 5,000 bootstrap samples of animals, keeping each animal's observations together.

## Recorded correction families

BH denotes Benjamini–Hochberg correction; the number states the family size. Correction families are not combined globally, and a family can span main and Extended Data panels.

| Analysis | Recorded tests and family |
|---|---|
| Figure 1 | Separate BH4 reward-acquisition, BH3 reward-context and BH6 first-visit route families. All six route contrasts are captioned, although only three significant brackets are drawn. |
| Figure 2 and ED9l | Successive-gap associations form one BH4 family: Figure 2g/l/q and ED9l. The fourth member is the displayed level-map association, not an unshown control. |
| Figure 3 and ED9n/o | Two separate BH4 experience-trend families: Figure 3f/i/l plus ED9n for early alignment, and Figure 3g/j/m plus ED9o for long-gap stability. |
| Figure 4 | Whole-maze and reward/control error tests and the panel-f behavioral association use raw p values. The two route-direction enrichment tests form BH2 using exact sign-flips of animal-mean effects. |
| Figure 5e | A nominal two-sided slope test with later exploration efficiency as response and level-map correlation as predictor, using 36 session pairs from 14 animals and standard errors clustered by animal. ED10a's level × exploration test uses the same direction with a different cohort and correction; see below. |
| ED1 | Natural-log outcomes with animal-clustered CR1 standard errors and two-sided t(G-1) tests. The three endpoints, three group contrasts and two displayed axes form BH18. |
| ED2 | First-30-min turning biases use separate BH4 families for the four biases in each day comparison, and BH3 for the three supported depth strata. |
| ED3 | BH4 across the four dwell-time contrasts in a–d. Whole-route scores use the same six-contrast BH6 family as Figure 1j; ED3f displays the depth-wise descriptive summaries. |
| ED5b–d | Means equally weight session pairs; confidence intervals use standard errors clustered by animal. All six pairwise contrasts between the four day combinations use exact sign-flips with all pairs of an animal flipped together, followed by BH6 within each neural metric. The displayed contrasts use these adjusted probabilities. |
| ED7a–c; d–f | Separate analysis scopes: three early-alignment age tests (place maps, PVs and ensemble rates; BH3), and four long-gap age tests (those measures plus level maps; BH4). Long-gap models use age at the later session and adjust for the cycle interval. The level-map long-gap age fit is tabulated rather than plotted in ED7. |
| ED5e | One maze-versus-outside comparison for Day i versus Day i+90; nominal p. The other three day combinations are not included in this context test. |
| ED6 fixed reference | BH12 across three neural measures and four displayed exposure definitions: 10, 20 and 30 minutes, and 3,000 steps in the maze. The 30-minute definition is shown in Figure 3. |
| ED6 sliding pivot | Full jointly available curves, with three separate BH3 families across the neural measures: Day i slopes, Day i+1 slopes and Day i+1 minus Day i differences in within-curve standard deviation. |
| ED8b–c | Panel b: one-sided exact sign-flip tests on animal-mean control-minus-reward errors, with BH4 over two source days × two sampling controls. Panel c: two-sided pair-level contrasts with standard errors clustered by animal, with BH2 over source-day variants. |
| ED8d–j | Route enrichment uses two-sided exact sign-flips of animal-mean effects. The two first-day directions in d share Figure 4e's BH2; the two second-day directions form a separate BH2. Reliability, equal-frame and starting-node contrasts in e/f/i form BH8. Rewarded-minus-timeout (g) and cross-direction enrichment (j) are separate nominal tests. The ten tests of panels e, f, g, i and j are listed in `figures/extended_data_08/data/route_statistics.csv`. |
| ED9a–c, g, i, k | Distribution, shuffle-null, partition, joint-correlation and Days-apart displays are descriptive, not a combined multiple-testing family. The mean-null comparisons in b/c use saved neuron-ID shuffle draws. |
| ED9d, f, h, j | d: nominal one-sided observed-minus-shuffled decoding test. f: BH2 over two one-sided preferred-level shift tests. h: BH2 over one-sided level-minus-horizontal/vertical contrasts. j: nominal one-sided observed-minus-within-level-shuffle test. These pair-level tests use standard errors clustered by animal. |
| ED10a | Each neural-measure row forms a BH3 family over its three behavioral outcomes. Stars use regression tests with standard errors clustered by animal, not ordinary Pearson tests. |
| ED10b | Three exact animal-score sign-flip tests with BH3: the reward-versus-level difference for each behavior and their cross-behavior difference. Commonality bars partition explained variance and are not themselves these profile tests. |
| ED10c | BH4 across the four displayed first-30-min turning biases. All four annotations use the same family. |

Figure 5e uses the 36-pair level-map cohort and a nominal p value. ED10a uses 30 pairs satisfying the common neural-behavior support required across its six neural readouts; its level × exploration regression also has exploration efficiency as response and level-map correlation as predictor, followed by row-wise BH3. The two panels differ in cohort and correction, so they should not be treated as duplicate statistical tests.

## Additional tests

The Figure 1f within-return experience trend uses 56 sessions in 17 animals and a nominal test. ED9f's two preferred-level transition tests are one-sided tests of increased adjacent-level frequency and reduced mean shift distance, using 36 pair observations with standard errors clustered by animal and BH2.

The [code guide](CODE.md) locates the calculation functions and their inputs. Raw-data processing, map fitting and decoder fitting are upstream and are not repeated here. See the [reproduction instructions](REPRODUCING.md) for running one figure or the complete package.
