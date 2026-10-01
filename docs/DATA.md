# Find and read the Source Data

Start from a figure or table linked on the [repository home page](../README.md). Each has the same structure:

```text
figure_02/
  README.md       what is shown and how to rebuild it
  build.py        runnable entry point
  data/           derived observations, summaries and examples
    README.md     linked file list
    index.csv     columns/formats, checksums and original source paths
  code/           frozen figure-specific helpers and shared-code links
```

The data files are physically in these folders, not just linked to an old nested archive. Repeated byte-identical helper/input copies are combined only when the retained file has the same SHA256. All original source paths can still be reconstructed.

## File formats

| Format | What to inspect |
|---|---|
| CSV / compressed CSV | Per-unit observations, plotted summaries, null values or supporting audit tables; use the column names and the associated builder to identify the unit |
| NPZ / NPY | Numerical arrays, example maps and saved simulation/shuffle containers; the builder documents field access |
| JSON | Structured numerical values or supporting metadata/configuration |
| MAT / FIG | Original example-map containers read by the frozen renderer |
| PNG / SVG / TIFF / PDF | Fixed example or illustrative assets, not the finished composite output used as a rendering shortcut |

[DATA_CATALOG.csv](DATA_CATALOG.csv) is the cross-figure file/column index. The individual `data/index.csv` files also retain original filenames and checksums. These are column/format inventories, not a replacement for the unit definitions in the analysis code. Do not interpret missing/blank values as zeros or merge different averaging units because two columns have similar names.

## Observations versus results

Some inputs are per-neuron, per-session-pair or per-mouse observations; others are frozen plot summaries or saved null/simulator draws. The [statistical replay](STATISTICS.md) recalculates the recorded analysis groups from their per-unit-derived inputs. It preserves their inclusion rules, clustering units and correction families; it does not refit upstream maps or decoders.

Session and animal IDs are retained. Source filenames containing an older figure number or round label are historical identifiers, not new figure assignments: the containing figure guide and build entry point determine their current use.
