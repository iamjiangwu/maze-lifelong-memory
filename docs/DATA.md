# Source Data

Each figure or table has a `data/` folder with the inputs used by its current builder and statistical analysis. [DATA_CATALOG.csv](DATA_CATALOG.csv) lists files across the repository; each `data/index.csv` records columns or array formats and checksums.

The package includes derived observations, plotted summaries, example maps and saved shuffle or simulation values—not raw recordings. Individual neuron, session-pair or animal values are retained for displayed observations and statistical tests. Figure 3's first-column curves use saved means, s.e.m. and support counts; its second- and third-column fits retain their individual observations. The package does not recalculate every descriptive curve from upstream measurements.

| Format | Contents |
|---|---|
| CSV / CSV.gz | Measurements, summaries, null draws and simulation results |
| NPZ / NPY | Numerical arrays and example maps |
| JSON | Structured measurements or settings |
| PNG / SVG / TIFF / PDF | Fixed illustrations or example assets |

Each data folder contains the inputs for the current panels and statistics. Correction families may span main and Extended Data panels; see the [statistics guide](STATISTICS.md). The technical mapping restores the build code's expected relative names during execution. Generated report and artwork names can retain internal suffixes; these do not identify additional datasets or analyses.

Animal/session IDs identify the retained observations. Some internal column names are retained for compatibility: `policy` refers to junction turning biases, `lb0` to first long bout time, and `rate` to ensemble rate measurements. Correlations between session representations and associations with behavior are different analyses; similar column names do not establish the statistical unit or correction family.

Each `data/README.md` identifies panel inputs and the observation unit. CSV empty numeric cells and numerical `NaN` values denote unavailable or undefined values, not zero. Explicit support/inclusion columns determine which observations enter each plot or test; do not replace excluded values with zero or treat every row as eligible. `n_pairs`, `n_animals` and neuron/bin counts are different denominators. Correlations, probabilities and efficiency ratios are dimensionless; time, distance and speed units are specified in the local guide or column name.

Session identifiers and selection hashes record provenance. Within one observation table, preserve animal identifiers when recalculating tests with standard errors clustered by animal. Numeric `q_*` columns store adjusted p values; columns ending `_family`, such as `q_bh4_family`, can instead contain the family's text identifier. The [statistics guide](STATISTICS.md) identifies the reported tests and families.
