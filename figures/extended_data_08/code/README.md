# Extended Data Figure 8: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `build_ed8_standardized_pv.py` draws decoder and route controls, forms animal means from retained pair inputs and recomputes the route comparisons. `accepted_schematic_primitives.py` supplies maze/route drawing. Internal panel keys differ from current display letters; use the [data crosswalk](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [accepted_schematic_primitives.py](accepted_schematic_primitives.py)
- [build_ed8_standardized_pv.py](build_ed8_standardized_pv.py)
