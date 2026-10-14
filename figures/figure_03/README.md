# Figure 3

Rebuild the figure with `python figures/figure_03/build.py` from the repository root.
The included observations reproduce within-cycle alignment and successive-cycle
similarity from Gaussian-smoothed S activity (1 s half-width at half-maximum).

`code/derived_panels.py` loads the included data, recalculates inference and
draws the complete page. `statistics(output_dir)` recomputes two four-test
experience families and the 12-test convergence family. No recording files
or upstream activity processing are required.

The first-window observations include 59 cycles from 22 animals; successive-cycle
observations include 37 pairs from 14 animals. Animal IDs define the repeated
observation clusters. The level-map rows are retained because they belong to
the four-test correction families, although this figure displays three measures.
