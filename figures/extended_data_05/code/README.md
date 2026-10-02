# Extended Data Figure 5: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_ed05_example_place_maps_assembly.py` draws the examples and comparisons. `recompute_ed05_pair_weighted_day_combinations.py` recalculates the equal-pair means, uncertainty and day-combination tests. `ed05_gallery_plotting.py` draws the supplied 32 examples from their saved normalized map arrays; it does not select neurons or reconstruct maps from recordings. See the [15 × 15 analysis versus 45 × 45 display distinction](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [ed05_gallery_plotting.py](ed05_gallery_plotting.py)
- [ed_supplement_common.py](ed_supplement_common.py)
- [recompute_ed05_pair_weighted_day_combinations.py](recompute_ed05_pair_weighted_day_combinations.py)
- [render_ed05_example_place_maps_assembly.py](render_ed05_example_place_maps_assembly.py)
