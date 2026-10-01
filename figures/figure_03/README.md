# Figure 3 — Within-session changes in spatial maps

Window-to-Day-2-template curves and experience-dependent spatial-map measures.

## Reproduce

Run from the repository root:

```powershell
python figures/figure_03/build.py --output-dir outputs/3
```

The output folder contains `main_figure_3/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [day2_template_group_curve_summary.csv](data/day2_template_group_curve_summary.csv)
- [combined_first30_points_all_metrics.csv](data/combined_first30_points_all_metrics.csv)
- [first_window_curve_point_alignment.csv](data/first_window_curve_point_alignment.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Window-to-Day-2-template means and per-pair curves, first-window observations and long-gap pair correlations. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
