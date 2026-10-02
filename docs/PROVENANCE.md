# Provenance

The included measurements are the inputs to the manuscript figures and tables. Current checksums and the mapping to the build code's expected paths are recorded in `reproduction/layout.json`. Reader-facing filenames do not denote separate analysis rounds. The current manifest identifies the authorized statistical revisions and the preceding source authority.

The launcher reconstructs the required input paths in a temporary directory. Figures use derived plotting values; statistical calculations use individual observations and saved shuffle/simulation values. Upstream raw-data processing is not part of this package.

The earlier complete source package is retained in Git history under `source-data-2026-09-30`. Generated validation reports are kept locally, outside this repository. Reproduction commands write reports to the selected output directory.

Original code uses the [MIT license](../LICENSE); original data and documentation use [CC BY 4.0](../DATA_LICENSE.md). Existing third-party notices take precedence.
