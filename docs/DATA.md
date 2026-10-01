# Source Data

Each figure or table has a `data/` folder with the inputs used by its current builder and statistical analysis. [DATA_CATALOG.csv](DATA_CATALOG.csv) lists files across the repository; each `data/index.csv` records columns or array formats and checksums.

The package includes derived observations, plotted summaries, example maps and saved shuffle or simulation values—not raw recordings. Individual neuron, session-pair or animal values are retained where required to recalculate uncertainty and tests. Figure 3, for example, includes per-animal/pair window values as well as its average curves and s.e.m.

| Format | Contents |
|---|---|
| CSV / CSV.gz | Measurements, summaries, null draws and simulation results |
| NPZ / NPY | Numerical arrays and example maps |
| JSON | Structured measurements or settings |
| MAT / FIG | Example-map containers |
| PNG / SVG / TIFF / PDF | Fixed illustrations or example assets |

Reader-facing filenames identify current inputs, not analysis rounds. The technical relocation map restores the build code’s expected names only during execution. Unused historical inputs and local validation reports are not part of the current package.

Animal/session IDs and numerical values are unchanged. Some internal column names are retained for compatibility: `policy` means junction turning biases, `lb0` means first long bout time, and `rate` denotes ensemble rate measurements. Statistical units and correction families are defined by the associated analysis, not inferred from similar column names. [Statistics guide](STATISTICS.md).
