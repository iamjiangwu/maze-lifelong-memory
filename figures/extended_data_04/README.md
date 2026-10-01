# Extended Data Figure 4 — Neuron registration across sessions

Derived registration features, field-of-view summaries and fixed example crops.

## Reproduce

Run from the repository root:

```powershell
python figures/extended_data_04/build.py --output-dir outputs/ed4
```

The output folder contains `extended_data_figure_4/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [ed04_registered_pair_features_long.csv](data/ed04_registered_pair_features_long.csv)
- [physical_fov_exact_36_source.csv](data/physical_fov_exact_36_source.csv)
- [ed04_candidate_features_directed_long.csv](data/ed04_candidate_features_directed_long.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Derived registration/FOV summaries and fixed example image crops; not acquisition movies. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
