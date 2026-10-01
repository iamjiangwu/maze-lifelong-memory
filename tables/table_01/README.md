# Supplementary Table 1 — Analysis and animal membership

Analysis-to-panel membership rows, mouse identifiers and count notes.

## Reproduce

Run from the repository root:

```powershell
python tables/table_01/build.py --output-dir outputs/table1
```

The output folder contains `supplementary_table_1/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [current_rows.json](data/current_rows.json)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Final analysis-to-panel-to-mouse membership rows and notes; mouse counts are reconstructed from ID lists. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Calculations

The table builder recalculates the membership counts or interval/cohort summaries from the included rows. The repository-wide [statistical replay](../../docs/STATISTICS.md) separately covers the figure analyses.
