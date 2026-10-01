# Extended Data Figure 5 — Day-combination and sampling controls

Place-map, population-vector and rate comparisons across four day combinations and contextual sampling controls.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_05/build.py --output-dir outputs/ed5
```

The output folder contains `extended_data_figure_5/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [ed05e_context_only_all_outside_pair_rows.csv](data/ed05e_context_only_all_outside_pair_rows.csv)
- [ed06_equal_time_pair_rows_corrected36.csv](data/ed06_equal_time_pair_rows_corrected36.csv)
- [ed05e_context_only_all_outside_clustered_stats.csv](data/ed05e_context_only_all_outside_clustered_stats.csv)
- [ed05e_context_only_all_outside_condition_means.csv](data/ed05e_context_only_all_outside_condition_means.csv)
- [ed05e_context_only_all_outside_pair_feasibility.csv](data/ed05e_context_only_all_outside_pair_feasibility.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Session-pair map/PV/rate values for four day combinations, contextual sampling comparisons and example maps. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
