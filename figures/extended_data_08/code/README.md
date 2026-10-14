# Reproduce Extended Data Figure 8

`derived_panels.py` loads derived observations, recalculates declared test families and draws the figure. Its public functions are `statistics(output_dir)` and `render(output_dir)`.

`current_science.py` contains the mathematical and drawing functions required by this figure. Shared `release_rendering` provides plotting, statistics and maze geometry. Page dimensions, palette, annotations and arrangement are explicit in code.

Balanced decoding averages pairs within animal and uses four exact one-sided sign-flip tests with BH4. PV recurrence uses animal-clustered BH2. Route intervals are Student-t intervals across mice. The eight tests in e, f and i form BH8; g and j each use one separate exact two-sided mouse sign-flip test.

Raw recordings, neural map estimation and decoder training are outside this release. Saved null draws and descriptive curve/band coordinates are Source Data.
