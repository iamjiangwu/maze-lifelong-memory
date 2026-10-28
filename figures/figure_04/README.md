# Figure 4. The reward location favors selective place stability

Rebuild the figure with `python figures/figure_04/build.py` from the repository root.
`code/derived_panels.py` recalculates statistical tests and draws the complete
page from included decoder errors, route profiles and illustration arrays.
No recording files or decoder fitting are required.

The main decoder counts binary native S events in trailing windows of 48
stored rows (1.92 s), with an occupancy prior. Coordinated behavior readouts
use 50 draws of 181 matched neurons and a flat prior. Decoder neurons meet
the 0.066 stored S units per retained minute gate in both sessions.
Route maps use unsmoothed S. Route inference averages pairs within each mouse
and corrects the two direction contrasts together.

The schematic is a separate illustration: C maps and traces, binary ticks
at rising trace samples, and a saved S-based posterior. Its posterior is not
recomputed from the illustrated ticks. Quantitative panels use native S events.
