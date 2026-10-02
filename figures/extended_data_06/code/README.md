# Extended Data Figure 6: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. `render_ed06_within_session_convergence_assembly.py` assembles fixed-reference and sliding-pivot curves from saved summaries and support masks. `fixed_stats_helpers.py` supplies fixed-reference statistical helpers and `ed_supplement_common.py` the drawing utilities. Pair-level slope/variability inputs are described in the [data guide](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [ed_supplement_common.py](ed_supplement_common.py)
- [fixed_stats_helpers.py](fixed_stats_helpers.py)
- [render_ed06_within_session_convergence_assembly.py](render_ed06_within_session_convergence_assembly.py)
