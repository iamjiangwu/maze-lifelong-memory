# Extended Data Figure 3: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_ed03_reward_path_fluency_assembly.py` assembles the four dwell comparisons, route schematic and depth-wise first choices. `fluency_panels.py` supplies the condition definitions and mouse-clustered confidence intervals for the displayed dwell means. Significance annotations come from the included statistics table. See the [data guide](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [render_ed03_reward_path_fluency_assembly.py](render_ed03_reward_path_fluency_assembly.py)
- [fluency_panels.py](fluency_panels.py)
