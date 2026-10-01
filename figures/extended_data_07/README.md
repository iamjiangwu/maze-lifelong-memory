# Extended Data Figure 7 — Age and experience

First-window and long-gap spatial-map observations with age/experience predictors.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_07/build.py --output-dir outputs/ed7
```

The output folder contains `extended_data_figure_7/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [ED07_FIRST30_PLOT_READY_MAINTAINED59.csv](data/ED07_FIRST30_PLOT_READY_MAINTAINED59.csv)
- [ed09_long_gap_age_joined_corrected36.csv](data/ed09_long_gap_age_joined_corrected36.csv)
- [combined_first30_points_all_metrics.csv](data/combined_first30_points_all_metrics.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: First-window observations with age/experience predictors and animal/session identities. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
