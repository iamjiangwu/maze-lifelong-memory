# main_figure_2

Neuron/pair correlations, map examples, registered-neuron metadata and saved shuffle draws; pair-level gap/recurrence inputs.

Numeric and fixed illustrative inputs are in `source/`. The frozen renderer and statistical-input provenance are included there. `accepted_outputs/` contains the latest approved output, used only AFTER execution to check reproduction.

From the release root: `python -B reproduce.py --artifact main_figure_2 --output-dir <fresh-directory>`. Use `--statistics-only` at the release root to regenerate all correction families (some span more than one figure).
