# Extended Data Figure 3 — Navigation after latent experience

Reward-route fluency and route fractions across behavioral conditions.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_03/build.py --output-dir outputs/ed3
```

The output folder contains `extended_data_figure_3/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [ed04_session_route_current_plus_new12.csv](data/ed04_session_route_current_plus_new12.csv)
- [fluency_plot_rows.csv](data/fluency_plot_rows.csv)
- [fluency_statistics.csv](data/fluency_statistics.csv)
- [ed04_route_events_current_plus_new12.csv](data/ed04_route_events_current_plus_new12.csv)
- [ed04_route_definition_current_plus_new12.csv](data/ed04_route_definition_current_plus_new12.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Session reward-route fluency and route fractions, with animal IDs and behavioral strata. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
