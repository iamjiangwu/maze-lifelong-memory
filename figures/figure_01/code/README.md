# Figure 1: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure1.py` assembles the example and quantitative panels; the timeline, trajectory and layout modules provide its drawing functions. They consume the files mapped in the [data guide](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_figure1_binary_tree_preview.py](build_figure1_binary_tree_preview.py)
- [build_figure1_ij_spacing.py](build_figure1_ij_spacing.py)
- [build_figure1_layout_alternatives.py](build_figure1_layout_alternatives.py)
- [figure1_complete_timeline_source.py](figure1_complete_timeline_source.py)
- [preview_figure1_assembly.py](preview_figure1_assembly.py)
- [rebuild_figure1.py](rebuild_figure1.py)
- [source_runtime.py](source_runtime.py)
- [trajectory_panel.py](trajectory_panel.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [figure_panels.py](../../../reproduction/library/figure_panels.py)
- [figure_panels__5697d9ffa6.py](../../../reproduction/library/figure_panels__5697d9ffa6.py)
- [preview_figure3_assembly__453e69706c.py](../../../reproduction/library/preview_figure3_assembly__453e69706c.py)
