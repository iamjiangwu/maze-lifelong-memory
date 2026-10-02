# Extended Data Figure 7: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_ed07_slide9_powerpoint_matched.py` draws and exports the six age-association panels from included observations and saved model coefficients; no slide deck or upstream analysis module is required. The shared launcher supplies the tables and palettes. The [data guide](../data/README.md) distinguishes age in days from recording experience and describes the model units.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [render_ed07_slide9_powerpoint_matched.py](render_ed07_slide9_powerpoint_matched.py)
