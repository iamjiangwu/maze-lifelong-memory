# Technical reproduction layer

Use the root [README](../README.md) to find figures and tables, or [reproduction instructions](../docs/REPRODUCING.md) to run them.

This directory contains the shared build machinery and source mapping:

- `layout.json` maps each required runtime input path to its repository path and current SHA256.
- `snapshot/RELEASE_MANIFEST.json` records the source inventory and all 17 artifact configurations.
- `snapshot/reproduce.py`, `snapshot/recompute_statistics.py` and `snapshot/shared/` contain the rendering and statistics engines.
- Figure-specific plotting/export helpers are in each figure's `code/` folder; shared plotting helpers are in `runtime/`.
- `runtime/` contains the shared Python support package as readable source files. The [code guide](../docs/CODE.md) maps each figure to its drawing functions.
- `metadata/` contains source inventories and settings consumed by the builders. For panel definitions and statistical interpretation, use the figure data guides and [statistics guide](../docs/STATISTICS.md).

The root launcher authenticates the manifest and checks every retained input against it. It restores the required paths in a temporary directory and writes a temporary release inventory for those files. The engines then run using this temporary source layout.

Generated outputs and run reports go to the selected output directory. Artwork build completion and statistical agreement are separate checks; see [reproduction instructions](../docs/REPRODUCING.md).
