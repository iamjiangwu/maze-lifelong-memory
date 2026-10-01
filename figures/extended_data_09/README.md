# Extended Data Figure 9 — Level maps and decoder controls

Level-map comparisons, neuron-ID shuffle/decoder controls, and gap/experience observations.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_09/build.py --output-dir outputs/ed9
```

The output folder contains `extended_data_figure_9/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [level_gap_pair_values.csv](data/level_gap_pair_values.csv)
- [first30_points.csv](data/first30_points.csv)
- [consecutive_points.csv](data/consecutive_points.csv)
- [cycles.csv](data/cycles.csv)
- [joint_cloud.csv](data/joint_cloud.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Level pair/neuron/axis values, saved shuffle draws, decoder comparisons, gap and experience inputs. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
