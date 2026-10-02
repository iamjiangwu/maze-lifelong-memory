# Extended Data Figure 5: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_ed05_example_place_maps_assembly.py` draws the examples and comparisons. `recompute_ed05_pair_weighted_day_combinations.py` recalculates the equal-pair means, uncertainty and day-combination tests. The example-map helper modules supply map drawing/selection functions; the build uses the supplied selected examples. See the [15 × 15 analysis versus 45 × 45 display distinction](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_ed06_example_place_map_candidates_v3.py](build_ed06_example_place_map_candidates_v3.py)
- [build_ed06_example_place_map_shortlist_32.py](build_ed06_example_place_map_shortlist_32.py)
- [ed_supplement_common.py](ed_supplement_common.py)
- [recompute_ed05_pair_weighted_day_combinations.py](recompute_ed05_pair_weighted_day_combinations.py)
- [render_ed05_example_place_maps_assembly.py](render_ed05_example_place_maps_assembly.py)
