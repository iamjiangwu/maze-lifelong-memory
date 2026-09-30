# extended_data_figure_10

Fixed-K166 neural/behavior pair rows; separate 36-pair policy rows and full 18-test policy/window family.

Numeric and fixed illustrative inputs are in `source/`. The frozen renderer and statistical-input provenance are included there. `accepted_outputs/` contains the latest approved output, used only AFTER execution to check reproduction.

From the release root: `python -B reproduce.py --artifact extended_data_figure_10 --output-dir <fresh-directory>`. Use `--statistics-only` at the release root to regenerate all correction families (some span more than one figure).
