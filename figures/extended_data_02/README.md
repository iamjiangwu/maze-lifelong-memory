# Extended Data Figure 2 — Turning policies and simulation controls

Turning-policy/depth observations, simulator outputs and matched-return comparisons.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_02/build.py --output-dir outputs/ed2
```

The output folder contains `extended_data_figure_2/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [fixed_time_policy_rows.csv](data/fixed_time_policy_rows.csv)
- [equal_junction_simulation_session_rows.csv](data/equal_junction_simulation_session_rows.csv)
- [equal_junction_depth_group_cluster_stats.csv](data/equal_junction_depth_group_cluster_stats.csv)
- [panel_g_depth_contrasts.csv](data/panel_g_depth_contrasts.csv)
- [panel_h_four_vs_depth_gap.csv](data/panel_h_four_vs_depth_gap.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Session turning-policy/depth values and fixed simulator outputs; matched-return effects for seeded mouse bootstrap. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
