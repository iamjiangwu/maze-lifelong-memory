# Figure 5: code

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure5.py` assembles the level-code schematic, neuron heatmaps, level-PV matrix, correlation distribution and exploration association. `level_heatmap_utils.py` supplies heatmap handling; shared renderers supply panel drawing. See the [data guide](../data/README.md) for the distinct neuron and session-pair units.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [level_heatmap_utils.py](level_heatmap_utils.py)
- [rebuild_figure5.py](rebuild_figure5.py)
- [SOURCE_REBUILD_MAIN_RUNTIME.zip](../../../reproduction/snapshot/shared/SOURCE_REBUILD_MAIN_RUNTIME.zip)
- [figure_panels__5697d9ffa6.py](../../../reproduction/library/figure_panels__5697d9ffa6.py)
- [preview_figure3_assembly__453e69706c.py](../../../reproduction/library/preview_figure3_assembly__453e69706c.py)
- [rebuild_current_main_figures.py](../../../reproduction/snapshot/shared/rebuild_current_main_figures.py)
- [render_main_figs_v3_publication__a7a324dc5c.py](../../../reproduction/library/render_main_figs_v3_publication__a7a324dc5c.py)
