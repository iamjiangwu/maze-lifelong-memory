# Figure 2: code

The supported entry point is [../build.py](../build.py), run from the repository root. `render_main_figs_v3_publication.py` assembles the example, correlation distributions, null means, Days-apart curves and consecutive-gap fits; `preview_figure2_assembly.py` and shared panel functions supply its component plots. The [data guide](../data/README.md) distinguishes neuron, bin, pair and triplet inputs.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [preview_figure2_assembly.py](preview_figure2_assembly.py)
- [render_main_figs_v3_publication.py](render_main_figs_v3_publication.py)
- [figure_panels.py](../../../reproduction/library/figure_panels.py)
