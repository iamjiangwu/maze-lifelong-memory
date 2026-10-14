# Reproduce Extended Data Figure 10

`derived_panels.py` loads derived observations, recalculates declared test families and draws the figure. Its public functions are `statistics(output_dir)` and `render(output_dir)`.

`current_science.py` contains the mathematical and drawing functions required by this figure. Shared `release_rendering` provides plotting, statistics and maze geometry. Page dimensions, palette, annotations and arrangement are explicit in code.

Matrix cells show Pearson r; significance uses animal-clustered CR1 slopes and BH3 across three outcomes within each neural row. Current values yield 12 significant cells. Commonality partitions pooled two-predictor OLS R². Three profile claims use exact animal-score sign flips with BH3. Policies use animal-clustered CR1 slopes and BH4; saved bootstrap bands are descriptive.

Raw recordings, neural map estimation and decoder training are outside this release. Saved null draws and descriptive curve/band coordinates are Source Data.
