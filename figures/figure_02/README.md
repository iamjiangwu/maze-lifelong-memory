# Figure 2 — Spatial representations across days

Place-map, population-vector and ensemble-rate comparisons, map examples and shuffle controls.

## Reproduce

Run from the repository root:

```powershell
python figures/figure_02/build.py --output-dir outputs/2
```

The output folder contains `main_figure_2/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [adjacent_day_ensemble_rate_pair_nulls.npz](data/adjacent_day_ensemble_rate_pair_nulls.npz)
- [adjacent_day_ensemble_rate_pair_values.csv](data/adjacent_day_ensemble_rate_pair_values.csv)
- [ensemble_rate_pair_nulls_audit.npz](data/ensemble_rate_pair_nulls_audit.npz)
- [ensemble_rate_pair_values.csv](data/ensemble_rate_pair_values.csv)
- [consecutive_gap_clean_long_gap_corrected_points.csv](data/consecutive_gap_clean_long_gap_corrected_points.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Neuron/pair correlations, map examples, registered-neuron metadata and saved shuffle draws; pair-level gap/recurrence inputs. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
