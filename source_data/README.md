# Reproduce the current version of record from Source Data

This is the current, tested 2026-09-30 deliverable: five main figures, ten Extended Data figures (132 panels), and two supplementary tables. It incorporates the latest local approved figure and table updates. Older dated records and their legacy registries are preserved, not overwritten.

## One command on this Windows computer

From this release folder, use a NEW output directory:

```powershell
.\rebuild_all.ps1 -OutputDir C:\Temp\ltm-record-rebuild
```

This verifies input hashes, rebuilds all 17 artifacts in disposable source-only stages, compares them with the accepted outputs, then recalculates all 53 analysis groups (including means/intervals and statistical test tables). Results are written to the directory you choose; the source release and old records are not edited.

For only a figure or only statistics:

```powershell
python -B reproduce.py --artifact extended_data_figure_10 --output-dir C:\Temp\ed10-rebuilt
python -B reproduce.py --statistics-only --output-dir C:\Temp\statistics-rebuilt
python -B reproduce.py --check-only
```

The statistical replay writes per-unit-derived estimates, standard errors, confidence intervals, raw and corrected p-values where applicable, plus comparison reports. Corrections retain the accepted families; for example ED10a is CR1 with BH3 across each neural-measure row, whereas Figure4's matching regression uses nominal CR1 p. ED5 uses session-pair-weighted averages/CR1 contrasts. Rendering uses frozen numerical plot inputs; the separate statistical replay recalculates and verifies those statistics from unit-level observations. It does not merely copy their saved p-values. This recreates the accepted record, rather than providing a new-data analysis pipeline.

## What to distribute

The adjacent `source_data_release_2026-09-30_source_only.zip` contains code, derived numerical Source Data, fixed illustrative assets, input manifests, and this guide. It omits all accepted composite artwork and validation-output copies. After extracting it, use:

```powershell
.\rebuild_all.ps1 -OutputDir C:\Temp\ltm-source-rebuild -NoCompare
```

`-NoCompare` does not skip input-hash or statistical-result checks; it only skips comparisons against the omitted composite reference figures/workbooks. The complete local folder keeps references in each artifact's `accepted_outputs/`, separated from rendering inputs.

## Source-data boundary

`source/` holds the sufficient derived values for the displayed figures and their statistical analysis: session/neuron/pair/mouse observations, already calculated decoder and map values, window summaries, and saved null/simulator draws. Raw recordings, tracking/video processing, neural extraction, registration, map/decoder fitting and acquisition-to-measurement alignment are upstream; those pipelines are not rerun here. Some registration metadata, example maps and fixed illustrative images are required to draw their panels. Fixed illustration pixels are not statistical observations.

This boundary is intentionally different from the separate raw/processed-data minimal repository. No external scientific data drive is accessed. Original repository reads and Python network connections are blocked in the source-only stages; this is an audited execution boundary, not an OS-level sandbox.

`DATA_DICTIONARY.csv` lists numeric source files and column names; each artifact has `SOURCE_MANIFEST.json` and a short README. Added statistical-input provenance is under `source/statistical_inputs/`. The frozen current review/table authority is recorded in `RELEASE_MANIFEST.json`.

## Software

The tested Python versions are in `requirements.txt` and `RUNTIME_ENVIRONMENT.json`. Native vector export currently requires Windows, Microsoft Edge and Arial. The XLSX builders additionally require Node.js and `@oai/artifact-tool` (available in the Codex bundled runtime on this computer). The wrapper locates that runtime automatically; elsewhere pass `-NodePath` and `-ArtifactToolModules` explicitly. These are software dependencies, not additional scientific data. Do not describe this tested Windows export stack as cross-platform without a new export test.

## Evidence

`verification/REPRODUCTION_REPORT.json`: all 17 artifacts pass. Figure raster pixels match exactly; PDF pages/text/rendering match; both workbooks have the same data/formulas/merges. PDF/SVG/XLSX binary hashes can differ due to export metadata or the recalculated provenance-manifest digest. Table2's differing digest is verified against its newly generated ledger manifest, not silently ignored.

`verification/statistics/STATISTICS_REPORT.json`: all 53 statistical analysis groups pass from the distributed derived inputs. Original correction families and statistical units are preserved. ED10 exact-test replay reads its source CSV with round-trip float precision to preserve boundary ties; no accepted statistic is changed.

This release updates only the local version-of-record package. It does not upload to Overleaf, change captions/Methods, synchronize Desktop PNGs, or change the separate minimal raw-data repository.
