# Extended Data Figure 4: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. The registration renderer draws field-of-view displacement curves, reads saved histogram counts for the candidate/matched-neuron distributions, and places the two supplied example images. `ed_supplement_common.py` supplies drawing/export utilities. The [data guide](../data/README.md) identifies which panels use images, displacement observations or histogram bins.

These panels are descriptive; they do not use inferential tests. The launcher redraws them from the included images, displacement values and histogram counts. [Environment/setup](../../../docs/REPRODUCING.md).

- [ed_supplement_common.py](ed_supplement_common.py)
- [render_ed04_registration_quality_assembly.py](../../../figures/extended_data_04/code/render_ed04_registration_quality_assembly.py)
