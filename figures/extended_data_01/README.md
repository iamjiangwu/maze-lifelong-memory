# Extended Data Figure 1 — Behavioral measures and speed

Session-level navigation measures, speed, endpoint visits and behavioral timelines.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_01/build.py --output-dir outputs/ed1
```

The output folder contains `extended_data_figure_1/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [PANEL_I_CYCLE_SUMMARY_ROUND21.csv](data/PANEL_I_CYCLE_SUMMARY_ROUND21.csv)
- [PANEL_A_TRAJECTORY_PLOT_DATA_ROUND21.csv.gz](data/PANEL_A_TRAJECTORY_PLOT_DATA_ROUND21.csv.gz)
- [SPEED_SUMMARY_ROUND15.csv](data/SPEED_SUMMARY_ROUND15.csv)
- [SPEED_SUMMARY_ROUND16.csv](data/SPEED_SUMMARY_ROUND16.csv)
- [SPEED_SUMMARY_ROUND17.csv](data/SPEED_SUMMARY_ROUND17.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Behavior-session endpoints, time/tile measures, speed, event sequences and derived curve observations. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
