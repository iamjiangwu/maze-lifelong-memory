# Figure 5 — Maze-level representations and exploration

Level-map vectors, cross-depth population comparisons and exploration associations.

## Reproduce

Run from the repository root:

```powershell
python figures/figure_05/build.py --output-dir outputs/5
```

The output folder contains `main_figure_5/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [level_map_vs_exploration_efficiency_primary_clean_rows.csv](data/level_map_vs_exploration_efficiency_primary_clean_rows.csv)
- [level_vectors_tree_depth7_corrected36_d1v1.csv](data/level_vectors_tree_depth7_corrected36_d1v1.csv)
- [level_pv_pair_matrices.csv](data/level_pv_pair_matrices.csv)
- [level_pv_matrix.csv](data/level_pv_matrix.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Matched neuron level vectors, pair cross-depth matrices and level-map/behavior pair rows. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
