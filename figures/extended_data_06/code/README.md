# Reproduce Extended Data Figure 6

`derived_panels.py` loads derived observations, recalculates declared test families and draws the figure. Its public functions are `statistics(output_dir)` and `render(output_dir)`.

`current_science.py` contains the mathematical and drawing functions required by this figure. Shared `release_rendering` provides plotting, statistics and maze geometry. Page dimensions, palette, annotations and arrangement are explicit in code.

The observational unit is a recording-cycle pair; the inference cluster is the animal. Fixed-reference slopes use BH12. Sliding-pivot SD differences and the two day-specific slopes each use BH3 across the three metrics. Displays average pairs within animal before group means and SEM.

Raw recordings, neural map estimation and decoder training are outside this release. Saved null draws and descriptive curve/band coordinates are Source Data.
