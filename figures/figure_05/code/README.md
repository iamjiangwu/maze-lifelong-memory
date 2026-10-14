# Figure 5 code

`derived_panels.py` loads maps and correlation tables, averages the level
population vector matrix across session pairs, and calls the drawing code.
`inference.py` relates later exploration efficiency to level map correlation
using the 36 pair observations, with standard errors clustered by animal.
`drawing.py` draws and assembles the five panels.

Run `python figures/figure_05/build.py --output-dir outputs/figure5`
from the repository root.
