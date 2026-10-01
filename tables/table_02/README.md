# Supplementary Table 2 — Animals, sessions and return intervals

Session dates, return intervals, cohort membership and nominal month assignments.

## Reproduce

Run from the repository root:

```powershell
python tables/table_02/build.py --output-dir outputs/table2
```

The output folder contains `supplementary_table_2/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [behavior_return_points.csv](data/behavior_return_points.csv)
- [calcium_long_gap_pairs_corrected36.csv](data/calcium_long_gap_pairs_corrected36.csv)
- [experiment_metafile.csv](data/experiment_metafile.csv)
- [publication_notes.json](data/publication_notes.json)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Derived return interval/cohort registries and session-date metadata; ledger, summaries and display months are recalculated. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Calculations

The table builder recalculates the membership counts or interval/cohort summaries from the included rows. The repository-wide [statistical replay](../../docs/STATISTICS.md) separately covers the figure analyses.
