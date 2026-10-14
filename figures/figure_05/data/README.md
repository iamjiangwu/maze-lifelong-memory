# Figure 5 inputs

- Panel a is drawn from the seven level maze geometry.
- `ordered_level_maps.csv` contains the normalized source and target maps
  in their displayed neuron order (panel b).
- `level_population_vectors.csv` contains all 49 level comparisons for each
  registered session pair. Panel c averages each comparison across pairs.
- `level_correlation_histogram.csv` contains observed and shuffled counts
  and densities for panel d; the neuron identity shuffles are not rerun.
- `stability_and_exploration.csv` contains the 36 session pairs used in
  panel e, with animal identity, level map correlation and later exploration
  efficiency. The animal IDs preserve the dependence among pairs.

Panel e's regression, animal clustered inference and 5,000 animal bootstrap
confidence band are calculated from those observations. The displayed maps
and correlations are derived measurements; this package does not fit them
from neural recordings.
