# Figure 1: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure1.py` assembles the example and quantitative panels; the trajectory and layout modules provide its drawing functions. `tree_geometry.py` defines the binary-tree maze geometry. The modules consume the files mapped in the [data guide](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_figure1.py](build_figure1.py)
- [configure_panel_geometry.py](configure_panel_geometry.py)
- [panel_layout.py](panel_layout.py)
- [behavior_panels.py](behavior_panels.py)
- [rebuild_figure1.py](rebuild_figure1.py)
- [source_runtime.py](source_runtime.py)
- [trajectory_panel.py](trajectory_panel.py)
- [tree_geometry.py](tree_geometry.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [figure_panels.py](../../../reproduction/runtime/ca_analysis/figure_panels.py)
