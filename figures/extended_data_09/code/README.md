# Extended Data Figure 9: code

The supported entry point is [../build.py](../build.py), run from the repository root. `rebuild_ed9.py` draws the 15-panel a–o figure from level-code measurements, saved null draws and summary curves. Its `CROSSWALK` and the [data guide](../data/README.md) map current panel letters to their inputs. Maps and decoders are not refitted.

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [rebuild_ed9.py](rebuild_ed9.py)
