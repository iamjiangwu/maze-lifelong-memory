# Reading the code

Start with the figure's `data/README.md`, then its code entry point. The root
[reproduce.py](../reproduce.py) checks input hashes, selects the requested
artifact and writes to a new output directory.

Figures 2–5 and Extended Data Figures 5–10 use three small roles:

- `code/derived_panels.py` loads the included observations, calls statistics
  and drawing functions, and assembles the page.
- `code/inference.py` calculates estimates, uncertainty and hypothesis tests.
- `code/drawing.py` draws the panels and places them on the page.

Some figures also have a small helper for a particular schematic or route
display. Shared numerical and export functions are in
[release_rendering](../reproduction/runtime/release_rendering/). No neural
session loader, map estimator or decoder fit is imported by these builders.

Figure 1 and Extended Data Figures 1–4 retain their numerical drawing routines
under each figure's `code/`. [extended_builds.py](../reproduction/extended_builds.py)
supplies the paths required by the Extended Data routines. The launcher places
their inputs in a temporary folder according to `loader_paths` in
[layout.json](../reproduction/layout.json); these are local input mappings,
not links to a research repository.

The behavioral statistics are in
[behavior_statistics.py](../reproduction/behavior_statistics.py) (Figure 1 and
ED1) and [policy_statistics.py](../reproduction/policy_statistics.py) (ED2 and
ED3). Neural figure builders expose a `statistics(output_dir)` function.
These functions calculate results from observations, not from saved p values.
Companion observations are included when a correction family spans figures.

Tables 1 and 2 calculate counts or intervals from their source tables and
write PDF and XLSX files. Table 3 typesets the supplied glossary definitions.
Use the [statistics guide](STATISTICS.md) for units, models and correction
families, and the [run guide](REPRODUCING.md) for commands.
