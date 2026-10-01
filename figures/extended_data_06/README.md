# Extended Data Figure 6 — Window-level map changes

Window curves, first-window observations and early-change/pivot comparisons.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_06/build.py --output-dir outputs/ed6
```

The output folder contains `extended_data_figure_6/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [ed07_current59_group_curve_summary.csv](data/ed07_current59_group_curve_summary.csv)
- [ed07_current59_group_curve_summary_50pct_gate_2026-07-14.csv](data/ed07_current59_group_curve_summary_50pct_gate_2026-07-14.csv)
- [ed07_pivot_experience_group_curve_summary_current59.csv](data/ed07_pivot_experience_group_curve_summary_current59.csv)
- [ed07_pivot_day_specific_early_slope_stats_current59.csv](data/ed07_pivot_day_specific_early_slope_stats_current59.csv)
- [ed07_pivot_experience_display_limits_current59.csv](data/ed07_pivot_experience_display_limits_current59.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Derived window curves and per-pair slopes/first-window values for CR1/BH18 and pivot CR1/BH9. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
