# Recalculate statistics

From the repository root, choose a new output directory:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay recalculates 53 recorded analysis groups from included derived observations and saved null or simulation draws. It writes `statistics/STATISTICS_REPORT.json`, regenerated numerical tables, comparison tables and `BUILD.log` inside the chosen directory. The report identifies each group and its result. Figures use the saved plotting inputs; statistical replay independently checks the recorded estimates and tests.

Statistical units and correction families differ by analysis:

| Analysis | Unit, summary and inference |
|---|---|
| Figure 1 behavior | Session observations; mouse-clustered CR1 tests. Reward-acquisition, reward-context and route contrasts have separate BH4, BH3 and BH6 families. |
| Figure 2 and ED9 gap curves | Equally weighted session-pair means with descriptive 95% Student-t intervals. These intervals are separate from mouse-clustered association tests. |
| Figure 3 experience | Separate first-window and long-gap CR1/BH4 trend families. Day 2 template curves average pairs within animal in each source cohort, then pool cohort means and variances to report mean ± SEM. |
| Figure 4 | The reward-error/behavior regression uses mouse-clustered CR1 with nominal p-values. Route summaries weight animals equally; exact sign-flip tests retain their specified families. |
| ED5 day combinations | Equally weighted session-pair means with mouse-clustered CR1 95% intervals. Paired contrasts use mouse-clustered CR1 with BH3 within each neural metric; the context control has a separate BH4 family. |
| ED10a | Pair-row Pearson associations; tests use mouse-clustered CR1. Displayed stars use BH3 across the three behavior outcomes within each neural-measure row. The replay also checks the separate BH6 values across neural measures within each outcome. |
| ED10b/c | Commonality and clustered profile tests remain separate from policy associations; policy-window tests retain their BH18 family. |

CR1 accounts for repeated observations from the same mouse and uses a Student-t reference with `G − 1` degrees of freedom, where `G` is the number of mice. Mean/SEM summaries, confidence intervals and hypothesis tests are distinct outputs; the correction families above are not combined into one global family.

The full calculations are in the frozen [main statistics](../reproduction/snapshot/shared/source_release_main_statistics.py), [Extended Data statistics](../reproduction/snapshot/shared/source_release_ed_statistics.py) and [additional statistics](../reproduction/snapshot/shared/source_release_root_statistics.py) modules. Raw-data processing, map fitting and decoder fitting are upstream of this repository. See [reproduction instructions](REPRODUCING.md) for software requirements and figure/table commands.
