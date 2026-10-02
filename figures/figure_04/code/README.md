# Figure 4: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `build_figure4_standardized_pv.py` draws the route panel and supplies the [mapped data](../data/README.md) to the other renderers: `render_figure4_day_source_comparison.py` draws the decoder comparisons and supported-position map, `render_main_figure4_panel_f_ed10_k166.py` draws the matched behavioral fit, and `render_main_figs_v3_publication.py` draws the method schematic. The shared main-figure driver composes the panel SVGs.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_figure4_standardized_pv.py](build_figure4_standardized_pv.py)
- [render_figure4_day_source_comparison.py](render_figure4_day_source_comparison.py)
- [render_main_figs_v3_publication.py](render_main_figs_v3_publication.py)
- [render_main_figure4_panel_f_ed10_k166.py](render_main_figure4_panel_f_ed10_k166.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [figure_panels.py](../../../reproduction/runtime/ca_analysis/figure_panels.py)
- [rebuild_current_main_figures.py](../../../reproduction/snapshot/shared/rebuild_current_main_figures.py)
