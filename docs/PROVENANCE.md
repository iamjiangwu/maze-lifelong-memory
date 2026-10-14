# Provenance

The included measurements support the manuscript figures and tables. The [input catalog](../reproduction/layout.json) records all 18 artifacts, the distributed files and their checksums. It also identifies the publication PDFs against which the package was checked.

Figures use derived plotting values; statistical calculations use individual observations and saved shuffle or simulation values. Some behavioral and table renderers load their inputs through a temporary directory containing only the selected artifact's inputs and shared plotting support. Upstream raw-data processing is not part of this package.

Reproduction requires no external research repository or network data source. The launcher verifies the distributed inputs before starting a build.

Reproduction commands write artwork, tables and calculation reports to the selected output directory. Git history records changes to the source package.

Original code uses the [MIT license](../LICENSE); original data and documentation use [CC BY 4.0](../DATA_LICENSE.md). Existing third-party notices take precedence.
