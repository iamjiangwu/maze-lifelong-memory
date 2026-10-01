# Figure 1 — Behavior across maze experience

Navigation, reward acquisition, exploration and example trajectories.

## Reproduce

Run from the repository root:

```powershell
python figures/figure_01/build.py --output-dir outputs/1
```

The output folder contains `main_figure_1/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [reward_acquisition_day1_month_points_current_plus_new12.csv](data/reward_acquisition_day1_month_points_current_plus_new12.csv)
- [long_bout_group_points_current_plus_new12.csv](data/long_bout_group_points_current_plus_new12.csv)
- [ed04_session_route_current_plus_new12.csv](data/ed04_session_route_current_plus_new12.csv)
- [exploration_first_chunk_current_plus_new_points.csv](data/exploration_first_chunk_current_plus_new_points.csv)
- [figure1_endnode_timelines.csv](data/figure1_endnode_timelines.csv)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Derived behavior-session points, route scores and plotted trajectories; fixed maze illustration. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
