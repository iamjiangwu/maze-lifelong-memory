# Extended Data Figure 5. Examples and controls for long gap neural stability

Panel a contains 32 registered neuron-pair examples from saved normalized 45 × 45 maps; correlations were calculated on the 15 × 15 analysis grid. Panels b–d compare four day combinations for PV, place maps and ensemble rates. Panel e compares outside-maze activity with equal-frame maze samples.

Pairs receive equal weight; repeated pairs are clustered by animal. Each metric has six paired day-combination contrasts. Exact animal sign flips and CR1 inference each receive BH6 within that metric. Panel e uses one animal-clustered maze-minus-outside comparison.

Run from the repository root:

```powershell
python figures/extended_data_05/build.py --output-dir outputs/ed5
```

The release starts at approved saved derived observations and display arrays. Neural values use the accepted S/HWHM 1 configuration and 0.066 eligibility gate where applicable. Metric-specific eligibility exceptions remain in the input tables. Missing values and support masks are preserved.

[Data](data/README.md) · [Code](code/README.md) · [Requirements](../../docs/REPRODUCING.md)
