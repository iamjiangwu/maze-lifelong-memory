# Figure 2 code

`derived_panels.py` loads the examples and metric tables.
`inference.py` fits successive gap associations with animal clustered
standard errors and applies BH4; it also prepares the descriptive curves.
The saved shuffle means are compared with the observed pair mean.
`drawing.py` renders the examples, distributions, days apart curves and
recurrence plots, then assembles the published page.

Run `python figures/figure_02/build.py --output-dir outputs/figure2`
from the repository root.
