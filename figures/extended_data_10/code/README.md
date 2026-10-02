# Extended Data Figure 10: code

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure.py` configures the input package; `render_ed10_slide12_powerpoint_matched.py` draws the three panels. `_ed10_neural_eligibility.py` preserves common support and `profile_helpers.py` calculates commonality components and profile tests. No slide deck is required. See the [data guide](../data/README.md) for readout transformations and correction families.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [_ed10_neural_eligibility.py](_ed10_neural_eligibility.py)
- [profile_helpers.py](profile_helpers.py)
- [rebuild_figure.py](rebuild_figure.py)
- [render_ed10_slide12_powerpoint_matched.py](render_ed10_slide12_powerpoint_matched.py)
- [SOURCE_REBUILD_ED_RUNTIME.zip](../../../reproduction/snapshot/shared/SOURCE_REBUILD_ED_RUNTIME.zip)
- [preview_figure3_assembly.py](../../../reproduction/library/preview_figure3_assembly.py)
- [rebuild_current_extended_data_figures.py](../../../reproduction/snapshot/shared/rebuild_current_extended_data_figures.py)
