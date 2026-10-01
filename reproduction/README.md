# Technical reproduction layer

Most readers can use the root [README](../README.md) and the individual figure/table folders.

This directory holds the frozen shared build machinery and the relocation map:

- `layout.json`: every original snapshot path, its current repository path and unchanged SHA256.
- `snapshot/RELEASE_MANIFEST.json`: the original sealed release inventory. Historical reference-artwork entries are retained as provenance; references are not distributed or used as inputs.
- `snapshot/reproduce.py` and `snapshot/recompute_statistics.py`: unchanged verified rendering/statistics engine.
- `snapshot/shared/` and `library/`: shared frozen helpers. Figure-specific helpers are under each figure's `code/`.
- `snapshot/artifact_manifests/` and `snapshot/artifact_notes/`: original artifact records and notes.

The original notes/guide describe the **old snapshot layout** and are preserved byte-for-byte as archival material. Use [docs/REPRODUCING.md](../docs/REPRODUCING.md) for the current repository commands. The original snapshot is also retained by the `source-data-2026-09-30` Git tag.

The root launcher checks full original-manifest coverage, validates all retained hashes, reconstructs the original relative paths in a temporary directory and calls the unchanged engine in source-only mode. This keeps readable navigation separate from the mathematical implementation. No original numerical value, estimator or correction family was changed by the reorganization.
