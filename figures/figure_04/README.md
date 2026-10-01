# Figure 4 — Reward decoding and navigation

Reward-location decoding, route profiles and neural–behavioral associations.

## Reproduce

Run from the repository root:

```powershell
python figures/figure_04/build.py --output-dir outputs/4
```

The output folder contains `main_figure_4/` with rebuilt artwork/workbooks and a build report. Use a new output directory for each run. [Software and output guide](../../docs/REPRODUCING.md).

## Inspect the data

- [panel_e_animal_profiles.csv](data/panel_e_animal_profiles.csv)
- [panel_e_curve.csv](data/panel_e_curve.csv)
- [spatial_decoder_example.json](data/spatial_decoder_example.json)
- [spatial_decoder_example.npz](data/spatial_decoder_example.npz)

[All data files and column names](data/README.md) · [Build code](code/README.md)

The data include: Displayed decoder/maps and per-pair errors; per-animal route profiles; fixed-K reward-decoder/behavior pair rows. These are derived Source Data, not raw recordings. Saved summaries and null/simulator draws remain part of the frozen reproduction inputs.

## Statistics

To recalculate the statistical outputs, run from the repository root:

```powershell
python reproduce.py --statistics-only --output-dir outputs/statistics
```

The replay runs all recorded analysis groups together because correction families can span panels. [Statistical units and correction families](../../docs/STATISTICS.md).
