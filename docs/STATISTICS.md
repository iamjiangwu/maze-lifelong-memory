# Statistical analyses

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay recalculates recorded analyses from derived observations and saved null or simulation draws. Regenerated tables and `statistics/STATISTICS_REPORT.json` are written to the chosen output directory. A passing comparison means agreement with the saved results, not confirmation that every manuscript statement uses the same method or correction family.

## Units and uncertainty

Unless specified otherwise, regressions use **standard errors clustered by animal**, with a finite-animal adjustment and a Student's t reference with `G − 1` degrees of freedom. The implementation calls this adjustment CR1. Pearson correlations, fitted slopes, confidence intervals and p values are distinct quantities; an ordinary Pearson p value is not the clustered regression p value.

Figure 2 and ED9 gap curves equally weight session-pair means and show descriptive 95% Student's t intervals. Figure 3's Day-2-reference summaries average pairs within animal in each source cohort, then pool cohort means and variances. For display, Figure 3e/h/k and ED9m separately apply a centred five-point moving average to the supplied mean and s.e.m.; bands show smoothed mean ± smoothed s.e.m. Figure 3 smooths before applying its support/time masks and requires `n_animals >= max(3, ceil(max_n / 2))`. ED9m filters `passes_support_mask` before smoothing, then draws points with at least three animals. These are display transformations, not recalculated s.e.m. values for smoothed individual trajectories.

Figure 3f/i/l and g/j/m use ordinary least-squares 95% confidence bands for the fitted mean, with n−2 degrees of freedom and rows treated as independent. These bands are descriptive: the stars instead use slope tests with standard errors clustered by animal and BH4. ED9n/o's fitted-mean bands instead use the animal-level sandwich covariance with the finite-sample adjustment and t(G−1) reference. Figure 4f's pointwise confidence intervals resample animals while keeping each animal's observations together.

## Recorded correction families

BH denotes Benjamini–Hochberg correction; the number states the family size. Correction families are not combined globally, and a family can span main and Extended Data panels.

| Analysis | Recorded tests and family |
|---|---|
| Figure 1 | Separate BH4 reward-acquisition, BH3 reward-context and BH6 first-visit route families. All six route contrasts are captioned, although only three significant brackets are drawn. |
| Figure 2 and ED9l | Successive-gap associations form one BH4 family: Figure 2g/l/q and ED9l. The fourth member is the displayed level-map association, not an unshown control. |
| Figure 3 and ED9n/o | Two separate BH4 experience-trend families: Figure 3f/i/l plus ED9n for early alignment, and Figure 3g/j/m plus ED9o for long-gap stability. Decoder sensitivity rows are excluded from these families. |
| Figure 4 | Whole-maze and reward/control error tests and the panel-f behavioral association use raw p values. The two route-direction enrichment tests form BH2 using exact sign-flips of animal-mean effects. |
| Figure 5e | A nominal two-sided slope test with level-map correlation as response and exploration efficiency as predictor, using 36 session pairs from 14 animals and standard errors clustered by animal. This differs from ED10a's level × exploration test in direction, cohort and correction; see below. |
| ED1 | The three endpoint comparisons, three group contrasts and two displayed axes form BH18. Decision-node endpoints are excluded. |
| ED2 | First-30-min turning biases use separate BH4 families for the four biases in each day comparison, and BH3 for the three supported depth strata. |
| ED3 | BH4 across the four dwell-time contrasts in a–d. Whole-route scores use the same six-contrast BH6 family as Figure 1j; ED3f displays the depth-wise descriptive summaries. |
| ED5b–d | Means equally weight session pairs; confidence intervals use standard errors clustered by animal. The displayed contrast tests use exact sign-flips with all pairs of an animal flipped together, then BH3 within each neural metric. The separately saved regression p values are not the plotted tests. |
| ED7a–c; d–f | Separate analysis scopes: three early-alignment age tests (place maps, PVs and ensemble rates; BH3), and four long-gap age tests (those measures plus level maps; BH4). The level-map long-gap age fit is tabulated rather than plotted in ED7; no level-map early-alignment age model is supplied. Ensemble-rate long-gap adjusted p = 0.066. |
| ED5e | One maze-versus-outside comparison for Day i versus Day i+90; nominal p. The other three day combinations are not included in this context test. |
| ED6 fixed reference | BH12 across three neural measures and four displayed exposure definitions: 10, 20 and 30 minutes, and 3,000 tile changes. The 30-minute definition is shown in Figure 3. Unshown 1,000- and 2,000-tile definitions are excluded. |
| ED6 sliding pivot | Full jointly available curves, with three separate BH3 families across the neural measures: Day 1 slopes, Day 2 slopes and Day 2 minus Day 1 differences in within-curve standard deviation. |
| ED8b–c | Panel b: one-sided exact sign-flip tests on animal-mean control-minus-reward errors, with BH4 over two source days × two sampling controls. Panel c: two-sided pair-level contrasts with standard errors clustered by animal, with BH2 over source-day variants. |
| ED8d–j | Route enrichment uses two-sided exact sign-flips of animal-mean effects. The two first-day directions in d share Figure 4e's BH2; the two second-day directions form a separate BH2. Reliability, equal-frame and starting-node contrasts in e/f/i form BH8. Rewarded-minus-timeout (g) and cross-direction enrichment (j) are separate nominal tests. |
| ED9a–c, g, i, k | Distribution, shuffle-null, partition, joint-correlation and Days-apart displays are descriptive, not a combined multiple-testing family. The mean-null comparisons in b/c use saved neuron-ID shuffle draws. |
| ED9d, f, h, j | d: nominal one-sided observed-minus-shuffled decoding test. f: BH2 over two one-sided preferred-level shift tests. h: BH2 over one-sided level-minus-horizontal/vertical contrasts. j: nominal one-sided observed-minus-within-level-shuffle test. These pair-level tests use standard errors clustered by animal. |
| ED10a | Each neural-measure row forms a BH3 family over its three behavioral outcomes. Stars use regression tests with standard errors clustered by animal, not ordinary Pearson tests. |
| ED10b | Three exact animal-score sign-flip tests with BH3: the reward-versus-level difference for each behavior and their cross-behavior difference. Commonality bars partition explained variance and are not themselves these profile tests. |
| ED10c | BH4 across the four displayed first-30-min turning biases. All four annotations use the same family. |

ED10b BH3 p values are 0.33 for the first-long-bout contrast and 0.013 for both exploration and the cross-behavior contrast.

Figure 5e uses the 36-pair level-map cohort. ED10a uses 30 pairs satisfying the common neural-behavior support required across its six neural readouts; its level × exploration regression has exploration efficiency as response and level-map correlation as predictor, followed by row-wise BH3. A Pearson correlation is symmetric, but a slope test with standard errors clustered by animal need not give the same p value after reversing response and predictor. The two panels therefore should not be treated as duplicate statistical tests.

## Additional tests

The Figure 1f within-return experience trend is replayed from 56 sessions in 17 animals (fold change per return cycle 0.71; nominal p = 0.036). ED9f's two preferred-level transition tests are replayed from 36 pair observations, with standard errors clustered by animal and BH2.

The calculations are in the [main](../reproduction/snapshot/shared/source_release_main_statistics.py), [Extended Data](../reproduction/snapshot/shared/source_release_ed_statistics.py) and [additional](../reproduction/snapshot/shared/source_release_root_statistics.py) statistics modules. Raw-data processing, map fitting and decoder fitting are upstream. [Reproduction instructions](REPRODUCING.md).
