# Extended Data Figure 10. Neural preservation and later behavior

Panels a/b use 30 successive long-gap pairs from 14 animals, 50 coordinated draws of 181 matched neurons and the accepted flat-prior decoder. Panel c uses 36 pairs and policies from the first 30 cumulative maze minutes of the later session.

Matrix cells show Pearson r; significance uses animal-clustered CR1 slopes and BH3 across three outcomes within each neural row. Current values yield 12 significant cells. Commonality partitions pooled two-predictor OLS R². Three profile claims use exact animal-score sign flips with BH3. Policies use animal-clustered CR1 slopes and BH4; saved bootstrap bands are descriptive.

Run from the repository root:

```powershell
python figures/extended_data_10/build.py --output-dir outputs/ed10
```

The release starts at approved saved derived observations and display arrays. Neural values use the accepted S/HWHM 1 configuration and 0.066 eligibility gate where applicable. Metric-specific eligibility exceptions remain in the input tables. Missing values and support masks are preserved.

[Data](data/README.md) · [Code](code/README.md) · [Requirements](../../docs/REPRODUCING.md)
