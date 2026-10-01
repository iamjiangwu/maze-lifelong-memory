# Technical reproduction layer

Use the root [README](../README.md) to find figures and tables, or [reproduction instructions](../docs/REPRODUCING.md) to run them.

This directory contains the shared build machinery and source mapping:

- `layout.json` maps each retained original input path to its current repository path and unchanged SHA256.
- `snapshot/RELEASE_MANIFEST.json` preserves the original sealed inventory and all 17 artifact configurations.
- `snapshot/reproduce.py`, `snapshot/recompute_statistics.py` and `snapshot/shared/` contain the frozen rendering and statistics engines.
- `library/` contains shared helper modules; figure-specific helpers are in each figure's `code/` folder.

The root launcher authenticates the original manifest and checks every retained input against it. It restores the retained paths in a temporary directory and writes a temporary release inventory for those files. Explicitly inspected checksum lists also omit entries for files no longer distributed; their retained entries and scientific settings remain unchanged. The engines then run using this temporary source layout.

The repository's original manifest and numerical inputs are preserved. Generated outputs go to the selected output directory. The original source-data snapshot remains recoverable through the `source-data-2026-09-30` Git tag.
