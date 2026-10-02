# Figure 3: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_figure.py` configures the bundled inputs and calls the Figure 3 renderer. It plots saved mean/s.e.m. curves in the first column and observation-level fits in the other two columns, as mapped in the [data guide](../data/README.md). It does not reconstruct individual moving-window curves.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [rebuild_figure.py](rebuild_figure.py)
- [render_main_figs_v3_publication.py](render_main_figs_v3_publication.py)
- [Shared runtime source](../../../reproduction/runtime/)
- [figure_panels.py](../../../reproduction/library/figure_panels.py)
- [rebuild_current_main_figures.py](../../../reproduction/snapshot/shared/rebuild_current_main_figures.py)
