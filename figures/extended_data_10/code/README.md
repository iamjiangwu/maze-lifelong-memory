# Extended Data Figure 10: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure.py` configures the input package; `render_ed10_slide12_powerpoint_matched.py` draws the three panels and `profile_helpers.py` calculates commonality components and profile tests. No slide deck is required. See the [data guide](../data/README.md) for readout transformations and correction families.

`correlation_statistics` calculates the correlation matrix's animal clustered tests and BH3 correction. The renderer saves those statistics and the displayed correlation values beside the artwork. Source checksums are verified by the launcher before rendering.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [profile_helpers.py](profile_helpers.py)
- [rebuild_figure.py](rebuild_figure.py)
- [render_ed10_slide12_powerpoint_matched.py](render_ed10_slide12_powerpoint_matched.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [rebuild_current_extended_data_figures.py](../../../reproduction/snapshot/shared/rebuild_current_extended_data_figures.py)
