# Extended Data Figure 4: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. The shared registration renderer draws field-of-view displacement curves and candidate/matched-neuron distributions, and places the two supplied example images. `ed_supplement_common.py` supplies drawing/export utilities. The [data guide](../data/README.md) identifies which panels use images or numerical observations.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [ed_supplement_common.py](ed_supplement_common.py)
- [render_ed04_registration_quality_assembly.py](../../../figures/extended_data_04/code/render_ed04_registration_quality_assembly.py)
