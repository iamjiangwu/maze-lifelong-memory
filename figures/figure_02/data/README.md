# Figure 2 inputs

Panel a uses the two local correlation image arrays and their color maps.
Panel b uses four matched neuron examples: `matched_neurons.csv` identifies
the sessions, neurons and displayed correlations; the compressed array holds
only their plotted trajectories and activity direction markers.

The three metric prefixes are `pv` (population vector), `place` (place map)
and `rate` (ensemble rate). Each has a schematic table, histogram counts and
densities, pair means and saved shuffle draws, and a days apart summary.
Saved shuffle draws are not regenerated.
The `*_gap.csv` files also contain five saved `shuffle_test` rows per metric.
These compare each pair's observed correlation with its neuron-ID shuffle mean,
using standard errors clustered by animal and BH correction across the five
displayed days-apart groups. The statistics command reproduces these tests.

`recurrence_points.csv` contains individual observations from successive
gaps, including the level map observations displayed in ED9l. All four metrics
are needed to recalculate their shared BH4 correction. Neurons or positions
are nested within animals; inference clusters observations by animal.
The rate relationship uses one observation per session chain.

The supplied neural measurements use deconvolved S smoothed in time with a
Gaussian half width at half maximum of 1 second. Map analyses apply the
specified S activity gate; ensemble rate retains its ungated definition.
The Source Data builder starts at these measurements, not neural sessions.

See [index.csv](index.csv) for array shapes and table columns.
