# Extended Data Figure 10 — Neural–behavioral relationships

Fixed-neuron-count associations, reward/level commonality and turning-policy comparisons.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_10/build.py --output-dir outputs/ed10
```

The output folder contains `extended_data_figure_10/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [CURRENT_FIXED_NEURON_PAIR_ROWS.csv](data/CURRENT_FIXED_NEURON_PAIR_ROWS.csv)
- [policy_window_pair_rows.csv](data/policy_window_pair_rows.csv)
- [REWARD_LEVEL_COMMONALITY.csv](data/REWARD_LEVEL_COMMONALITY.csv)
- [REWARD_LEVEL_PROFILE_TESTS.csv](data/REWARD_LEVEL_PROFILE_TESTS.csv)
- [level_policy_bridge_join_corrected36.csv](data/level_policy_bridge_join_corrected36.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Fixed-K166 neural/behavior pair rows; separate 36-pair policy rows and full 18-test policy/window family. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
