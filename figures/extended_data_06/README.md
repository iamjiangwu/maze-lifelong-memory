# Extended Data Figure 6. Within-session neural stability

Fixed-reference columns compare moving Day i windows with the complete Day i+1 session. Sliding-pivot columns compare adjacent 30-minute windows within each day. Saved curves preserve support masks and gaps.

The observational unit is a recording-cycle pair; the inference cluster is the animal. Fixed-reference slopes use BH12. Sliding-pivot SD differences and the two day-specific slopes each use BH3 across the three metrics. Displays average pairs within animal before group means and SEM.

Run from the repository root:

```powershell
python figures/extended_data_06/build.py --output-dir outputs/ed6
```

The release starts at approved saved derived observations and display arrays. Neural values use the accepted S/HWHM 1 configuration and 0.066 eligibility gate where applicable. Metric-specific eligibility exceptions remain in the input tables. Missing values and support masks are preserved.

[Data](data/README.md) · [Code](code/README.md) · [Requirements](../../docs/REPRODUCING.md)
