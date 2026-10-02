# Provenance

The included measurements support the manuscript figures and tables. Checksums and the mapping between repository files and their build paths are recorded in [reproduction/layout.json](../reproduction/layout.json). The [release manifest](../reproduction/snapshot/RELEASE_MANIFEST.json) identifies the source inventory and artifact configurations.

The launcher reconstructs the required input paths in a temporary directory. Figures use derived plotting values; statistical calculations use individual observations and saved shuffle/simulation values. Upstream raw-data processing is not part of this package.

During reproduction, a Python-level guard blocks reads of unstaged source files in the repository checkout and Python network access. Installed Python libraries and output files remain accessible. This is not an operating-system sandbox.

Reproduction commands write artwork, tables and calculation reports to the selected output directory. Git history records changes to the source package.

Original code uses the [MIT license](../LICENSE); original data and documentation use [CC BY 4.0](../DATA_LICENSE.md). Existing third-party notices take precedence.
