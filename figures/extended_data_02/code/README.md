# Extended Data Figure 2: code

Run [../build.py](../build.py) through the repository's supported launcher. The
build reads the included first-30-minute policy tables and saved biased-walk
simulation results; it does not refit policies or rerun stochastic walks.
See the [panel/data mapping](../data/README.md) and
[environment/setup guide](../../../docs/REPRODUCING.md).

The current implementation has three modules:

- [render_ed2.py](render_ed2.py): loads the derived inputs, draws each of the nine
  final panels once, places the model labels in panel h, and assembles the page.
- [ed2_panels.py](ed2_panels.py): panel drawing, display transforms, table loading
  and cohort/support checks. Start with `build_panel_a`, `build_policy_scatter`,
  `build_panel_d` through `build_panel_h`, and `build_calibration` for f/i. The
  named private helpers apply axes, typography, marker and label placement.
- [ed2_export.py](ed2_export.py): places panel PDFs and the shared legend using
  the explicit page frames, then exports the vector page and raster formats.

The panel-i depth-varying calibration uses the same all-depth-strata support as
g/h: at least 10 P_BS events in each shallow, middle and deep stratum. Its
Naive/Return/Second-day cohort counts are 22/58/74; the four-policy calibration
in f retains 23/58/74. `depth_supported`, `validate_source_cohorts` and
`prepare_calibration` enforce these contracts. All windows are the first 30
cumulative TimeInMaze minutes; Explore-mode junction decisions define policies.

Recompute the figure's inferential results with the other recorded analyses via
`python reproduce.py --statistics-only --output-dir outputs/statistics`.
The [code reading map](../../../docs/CODE.md) identifies the statistical replay
entry points. No intermediate full-figure rendering or revision-specific
import chain is needed by this builder.
