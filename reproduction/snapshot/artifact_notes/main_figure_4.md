# main_figure_4

Displayed decoder/maps and per-pair errors; per-animal route profiles; fixed-K reward-decoder/behavior pair rows.

Numeric and fixed illustrative inputs are in `source/`. The frozen renderer and statistical-input provenance are included there. `accepted_outputs/` contains the latest approved output, used only AFTER execution to check reproduction.

From the release root: `python -B reproduce.py --artifact main_figure_4 --output-dir <fresh-directory>`. Use `--statistics-only` at the release root to regenerate all correction families (some span more than one figure).
