# Figure 4: code

The supported entry point is [../build.py](../build.py), run from the repository root. `build_figure4_standardized_pv.py` supplies the route panels, `render_figure4_day_source_comparison.py` the decoder comparisons, and `render_main_figure4_panel_f_ed10_k166.py` the matched behavioral fit. The publication renderer assembles the panels from the [mapped data](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_figure4_standardized_pv.py](build_figure4_standardized_pv.py)
- [render_figure4_day_source_comparison.py](render_figure4_day_source_comparison.py)
- [render_main_figs_v3_publication.py](render_main_figs_v3_publication.py)
- [render_main_figure4_panel_f_ed10_k166.py](render_main_figure4_panel_f_ed10_k166.py)
- [SOURCE_REBUILD_MAIN_RUNTIME.zip](../../../reproduction/snapshot/shared/SOURCE_REBUILD_MAIN_RUNTIME.zip)
- [figure_panels.py](../../../reproduction/library/figure_panels.py)
- [rebuild_current_main_figures.py](../../../reproduction/snapshot/shared/rebuild_current_main_figures.py)
