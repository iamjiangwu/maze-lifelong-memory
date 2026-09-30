# main_figure_5

Matched neuron level vectors, pair cross-depth matrices and level-map/behavior pair rows.

Numeric and fixed illustrative inputs are in `source/`. The frozen renderer and statistical-input provenance are included there. `accepted_outputs/` contains the latest approved output, used only AFTER execution to check reproduction.

From the release root: `python -B reproduce.py --artifact main_figure_5 --output-dir <fresh-directory>`. Use `--statistics-only` at the release root to regenerate all correction families (some span more than one figure).
