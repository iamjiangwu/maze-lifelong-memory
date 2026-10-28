# Extended Data Figure 7. Age associations with map stability

The top row relates first-30-minute Day i similarity to Day i age. The bottom row relates long-gap similarity to destination-session age. Lines use a common age coefficient with separate categorical experience or transition intercepts.

OLS uses age per 30 days, categorical adjustments, animal-clustered CR1 uncertainty and t(G−1) inference. The within-cycle family contains three metrics; long-gap BH4 also includes level maps, whose observations are retained in level_age_family_points.csv.

Run from the repository root:

```powershell
python figures/extended_data_07/build.py --output-dir outputs/ed7
```

The release starts at approved saved derived observations and display arrays. Neural values use the accepted S/HWHM 1 configuration and 0.066 eligibility gate where applicable. Metric-specific eligibility exceptions remain in the input tables. Missing values and support masks are preserved.

[Data](data/README.md) · [Code](code/README.md) · [Requirements](../../docs/REPRODUCING.md)
