# Figure 5: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure5.py` assembles the level-code schematic, neuron heatmaps, level-PV matrix, correlation distribution and exploration association. `level_heatmap_utils.py` supplies heatmap handling; shared renderers supply panel drawing. See the [data guide](../data/README.md) for the distinct neuron and session-pair units.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [level_heatmap_utils.py](../../../reproduction/runtime/ca_analysis/level_heatmap_utils.py)
- [rebuild_figure5.py](rebuild_figure5.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [figure_panels.py](../../../reproduction/runtime/ca_analysis/figure_panels.py)
- [plot_panels.py](plot_panels.py)
- [rebuild_current_main_figures.py](../../../reproduction/snapshot/shared/rebuild_current_main_figures.py)
- [export_figure.py](export_figure.py)
