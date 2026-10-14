# Reproduce Extended Data Figure 9

`derived_panels.py` loads derived observations, recalculates declared test families and draws the figure. Its public functions are `statistics(output_dir)` and `render(output_dir)`.

`current_science.py` contains the mathematical and drawing functions required by this figure. Shared `release_rendering` provides plotting, statistics and maze geometry. Page dimensions, palette, annotations and arrangement are explicit in code.

Pair-level tests use animal-clustered CR1 uncertainty. Preferred-level and partition comparisons each use BH2. Recurrence and both experience analyses retain full four-metric observation families in companion tables and recompute BH4. Null distributions are saved derived draws.

Raw recordings, neural map estimation and decoder training are outside this release. Saved null draws and descriptive curve/band coordinates are Source Data.
