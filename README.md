# Maze lifelong memory: figure Source Data

Derived Source Data and frozen build code for the approved 2026-09-30 version of record: **5 main figures, 10 Extended Data figures (132 panels), and 2 supplementary tables**. The included observations also reproduce **53 statistical analysis groups**.

This repository is the figure/statistics source-data deliverable, not the separate raw-data analysis repository. Raw calcium recordings, videos, acquisition processing, registration and map/decoder fitting are upstream and are not rerun here. Finished composite figures and workbooks are not included as build inputs.

## Reproduce

On the tested Windows environment, clone this repository, then run from its root using a new output directory:

```powershell
cd source_data
.\rebuild_all.ps1 -OutputDir C:\Temp\maze-source-rebuild -NoCompare
```

This checks the frozen input hashes, rebuilds all 17 figures/tables and recalculates their statistical outputs from included derived observations. `-NoCompare` skips only comparisons with finished reference artwork omitted from this repository. Input integrity and statistical-result checks still run.

To check inputs or run only statistics:

```powershell
python -B reproduce.py --no-compare --check-only
.\rebuild_all.ps1 -OutputDir C:\Temp\maze-statistics-rebuild -NoCompare -StatisticsOnly
```

For one artifact:

```powershell
.\rebuild_all.ps1 -OutputDir C:\Temp\maze-ed10-rebuild -NoCompare -Artifact extended_data_figure_10
```

See [the full reproduction guide](source_data/README.md) for the source boundary, statistical units/correction families, software dependencies and output reports.

## Software requirements

Python packages and tested versions are recorded in [requirements.txt](source_data/requirements.txt) and [RUNTIME_ENVIRONMENT.json](source_data/RUNTIME_ENVIRONMENT.json). Native vector exports require **Windows, Microsoft Edge and Arial**. The editable XLSX builders additionally require **Node.js and `@oai/artifact-tool`**, available in the Codex bundled runtime on the tested computer. The PowerShell wrapper detects that runtime there; on another computer provide `-NodePath` and `-ArtifactToolModules` explicitly. These are software dependencies, not missing scientific data. A plain Python installation alone is not sufficient to build every export format.

## Contents

- `source_data/main_figure_1/` through `main_figure_5/`: source inputs and per-figure instructions.
- `source_data/extended_data_figure_1/` through `extended_data_figure_10/`: Extended Data inputs and instructions.
- `source_data/supplementary_table_1/` and `supplementary_table_2/`: source-backed table builders and inputs.
- [DATA_DICTIONARY.csv](source_data/DATA_DICTIONARY.csv): numerical source files and column names.
- [RELEASE_MANIFEST.json](source_data/RELEASE_MANIFEST.json): frozen source inventory, checksums and figure/table authority.
- `docs/validation/`: executed validation receipts, not rendering inputs.
- [PROVENANCE.md](docs/PROVENANCE.md): package origin, licensing status and versioning boundary.

The frozen release manifest also records the omitted reference/verification files with explicit roles. Use `-NoCompare` for this source-only repository; do not modify the frozen manifest merely to remove those historical comparison entries. Original absolute paths in provenance are historical labels, not required data locations.

## Verified scope

All 17 artifacts were rebuilt using source-only stages without reads from the original scientific repository. Raster pixels and rendered PDF pages/text matched the approved exports; both workbooks matched their data, formulas and merges. All 53 statistical groups passed a separate derived-observation replay. The source-only sharing archive was also extracted and verified independently. Receipts: [artwork/table reproduction](docs/validation/REPRODUCTION_REPORT.json), [statistics](docs/validation/STATISTICS_REPORT.json), and [source-only archive](docs/validation/ARCHIVE_VALIDATION.json).

These are the executed local validation results for this snapshot, not a claim that every operating system or a newly installed environment has been tested. No DOI or manuscript-publication identifier has been assigned by preparing this repository.

## Access and reuse

No new code or data licence is granted by this repository preparation. An author-approved licence and publication citation remain to be supplied before public distribution; retain any notices already present in individual source files.
