# Source Data

Each figure or table has a `data/` folder with the inputs used by its current builder and statistical analysis. [DATA_CATALOG.csv](DATA_CATALOG.csv) lists files across the repository; each `data/index.csv` records columns or array formats and checksums.

The package includes derived observations, plotted summaries, example maps and saved shuffle or simulation values—not raw recordings. Individual neuron, session-pair or animal values are retained where required to recalculate uncertainty and tests. Figure 3, for example, includes per-animal/pair window values as well as its average curves and s.e.m.

| Format | Contents |
|---|---|
| CSV / CSV.gz | Measurements, summaries, null draws and simulation results |
| NPZ / NPY | Numerical arrays and example maps |
| JSON | Structured measurements or settings |
| PNG / SVG / TIFF / PDF | Fixed illustrations or example assets |

Reader-facing filenames identify the retained inputs, not analysis rounds. Correction families may span main and Extended Data panels; see the [statistics guide](STATISTICS.md). The technical relocation map restores the build code’s expected names only during execution; local validation reports are not included.

Animal/session IDs identify the retained observations. Some internal column names are retained for compatibility: `policy` refers to junction turning biases, `lb0` to first long bout time, and `rate` to ensemble rate measurements. Correlations between session representations and associations with behavior are different analyses; similar column names do not establish the statistical unit or correction family.
