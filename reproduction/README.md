# Technical reproduction layer

Use the root [README](../README.md) to find figures and tables, or [reproduction instructions](../docs/REPRODUCING.md) to run them.

This directory contains the shared build machinery and source mapping:

- `layout.json` maps each required runtime input path to its repository path and current SHA256.
- `snapshot/RELEASE_MANIFEST.json` seals the current source inventory, revision provenance and all 17 artifact configurations.
- `snapshot/reproduce.py`, `snapshot/recompute_statistics.py` and `snapshot/shared/` contain the frozen rendering and statistics engines.
- `library/` contains shared helper modules; figure-specific helpers are in each figure's `code/` folder.
- `runtime/` contains the shared Python support package as readable source files. The [code guide](../docs/CODE.md) maps each figure to its drawing functions.
- `metadata/` retains historical text needed by build contracts. It is not the current figure guide; some descriptions predate the displayed panels.

The root launcher authenticates the manifest and checks every retained input against it. It restores the required paths in a temporary directory and writes a temporary release inventory for those files. The engines then run using this temporary source layout.

Current definitions and known differences are listed in the [statistics guide](../docs/STATISTICS.md). Generated outputs go to the selected output directory. The original source-data snapshot remains recoverable through the `source-data-2026-09-30` Git tag; revised statistics are identified explicitly in the current manifest.
