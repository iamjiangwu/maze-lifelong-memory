# Source Data for Extended Data Figure 8

a: supported cohort/bin display values and support counts; b/c: pair observations; d/e/f/g/i: mouse-level observations; j: mouse-level route-bin profiles. The route schematic uses explicit geometry in code. Route controls use unsmoothed S (HWHM 0), unlike the S/HWHM 1 maps used for the spatial decoder and PV controls.

Panel c also includes saved recurrence tests and matched-source display means
and confidence intervals. Its pair values use frame-balanced reward/control
locations. In e, f, g, i and j, `record_type` separates `animal` observations
from the saved `statistic` rows; only animal rows enter recalculation.

Balanced decoding averages pairs within animal and uses four exact one-sided sign-flip tests with BH4. PV recurrence uses standard errors clustered by animal and BH correction across two tests. Route intervals are Student-t intervals across mice. The eight tests in e, f and i form BH8; g and j each use one separate exact two-sided sign-flip test on animal means.

`record_type` (or `row_type`) identifies observations, display summaries and inference rows. IDs identify animals, sessions, pairs, neurons or chains. Correlations and policy probabilities are dimensionless; decoder error is in bins. Undefined values are empty cells. Descriptive coordinates are supplied directly; statistical tests are recalculated from observations.
