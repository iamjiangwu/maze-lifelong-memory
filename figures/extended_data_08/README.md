# Extended Data Figure 8 — Reward-region and route controls

Balanced decoder comparisons, population-vector contrasts and route-bin observations.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_08/build.py --output-dir outputs/ed8
```

The output folder contains `extended_data_figure_8/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [accepted_b.csv](data/accepted_b.csv)
- [accepted_c.csv](data/accepted_c.csv)
- [accepted_routes.csv](data/accepted_routes.csv)
- [accepted_h.csv](data/accepted_h.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Balanced decoder mouse values, unstandardized PV pair contrasts and standardized route-bin/pair/mouse values. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
