# Extended Data Figure 1: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_extended_data_figure1.py` draws the numerical panels; `render_ed01_no_trajectories_source.py` assembles the current ten-panel a–j layout. Source filenames can retain earlier panel letters; use the current crosswalk in the [data guide](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [render_ed01_no_trajectories_source.py](render_ed01_no_trajectories_source.py)
- [render_extended_data_figure1.py](render_extended_data_figure1.py)
