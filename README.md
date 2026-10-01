# Maze lifelong memory

This project studies how mice learn a complex maze, remember it over months, and how their navigation relates to hippocampal neural representations. This repository contains the **derived Source Data and code for the manuscript's 5 main figures, 10 Extended Data figures and 2 supplementary tables**.

The current release starts from the measurements underlying the figures—not from raw recordings. It lets readers inspect those measurements, rebuild the artwork and tables, and recalculate the recorded statistical results.

[Background](#scientific-background) · [Data](#what-data-are-included) · [Code](#what-the-code-does) · [Reproduce](#reproduce-the-results) · [Figure guide](#find-a-figure-or-table)

## Scientific background

The central questions are how experience changes navigation through a complex environment, which neural representations persist across long intervals, and how those representations relate to behavior when animals return.

Mice navigate a **15 × 15-grid, seven-level binary-tree labyrinth with up to 64 endpoints**. Exploration is self-paced rather than organized into fixed experimenter-paced trials. Behavioral measurements and hippocampal calcium-imaging analyses are compared across initial learning, repeated experience and returns separated by long gaps. Different-session neural comparisons use registered neurons; the figures' inclusion rules and averaging units are retained in the supplied source tables and analysis code.

The analyses distinguish several aspects of the neural representation: two-dimensional place maps, population-vector similarity, ensemble activity rates, maze-level maps, and position decoding around the reward location. Behavioral measures include reward-acquisition timing, route use, exploration efficiency and approach speed. The repository contains the measurements and controls used to compare these representations and examine their relationships with navigation; it does not imply that every analysis uses the same animals, sessions or statistical unit.

## What data are included?

The data are organized by the figure or table that uses them. Each folder has a short guide, a linked file list, and an `index.csv` describing columns/formats and preserving the original source paths and checksums.

| Data | Examples and where to look |
|---|---|
| Behavioral observations | Session-level reward-acquisition, route and exploration measurements, speed summaries and plotted trajectories: [Figure 1](figures/figure_01/data/README.md), [ED1](figures/extended_data_01/data/README.md) |
| Neural map measurements | Per-neuron/per-pair correlations, example map arrays, level vectors and population summaries: [Figure 2](figures/figure_02/data/README.md), [Figure 3](figures/figure_03/data/README.md), [Figure 5](figures/figure_05/data/README.md) |
| Decoder and neural–behavioral measurements | Decoder errors, route profiles and matched neural/behavioral pair rows: [Figure 4](figures/figure_04/data/README.md), [ED10](figures/extended_data_10/data/README.md) |
| Controls and reference distributions | Saved shuffle/null and simulator outputs, sampling controls and derived control observations: [ED2](figures/extended_data_02/data/README.md), [ED5](figures/extended_data_05/data/README.md), [ED8](figures/extended_data_08/data/README.md), [ED9](figures/extended_data_09/data/README.md) |
| Animal/session information | Analysis membership, animal identifiers, session dates, return intervals and cohort assignments: [Table 1](tables/table_01/data/README.md), [Table 2](tables/table_02/data/README.md) |

For a concrete example, [ED10's matched-pair table](figures/extended_data_10/data/CURRENT_FIXED_NEURON_PAIR_ROWS.csv) contains animal/pair identifiers, neural map/decoder measurements, behavioral outcomes and support fields. It is a derived analysis table, not a calcium recording. Other figures may instead use neuron-level observations, session curves or animal means: rows should not be pooled across these units without following the associated analysis.

Files include **CSV/compressed CSV** tables, **NPZ/NPY** numerical arrays, **JSON** values/metadata, original example-map containers, and fixed illustrative assets. [The data guide](docs/DATA.md) explains these formats; [the cross-figure catalog](docs/DATA_CATALOG.csv) helps locate a particular file or column. Both individual observations and frozen plotting summaries are included where the reproduction requires them.

## What the code does

There are two related workflows:

1. **Figure/table building:** read the included numerical plotting inputs and example assets, then produce the manuscript artwork, editable workbooks and supporting table ledgers.
2. **Statistical replay:** recalculate the recorded descriptive summaries and inferential results from derived per-unit observations and saved null/simulation draws. This covers 53 grouped computations, preserving the original statistical units, clustering rules and multiple-testing families.

The code is accessible at three levels:

| Entry point | Purpose |
|---|---|
| [reproduce.py](reproduce.py), or [reproduce.ps1](reproduce.ps1) | Select a figure/table, verify input integrity and run the build or statistical replay |
| Each figure/table's `build.py` and `code/` guide | Find and run its figure-specific implementation; see [Figure 2's code](figures/figure_02/code/README.md) or [ED10's code](figures/extended_data_10/code/README.md) |
| [Statistical implementations](reproduction/snapshot/shared/) | Inspect the calculations in `source_release_main_statistics.py`, `source_release_ed_statistics.py` and `source_release_root_statistics.py`; [analysis index](docs/STATISTICS.md) |

Shared frozen helpers live in `reproduction/`. The launcher reconstructs their original input/import paths in a temporary directory so the public folders can be easy to browse without rewriting the verified scientific calculations. Readers do not need to assemble those paths manually.

## Reproduce the results

First use the [supported software environment](docs/REPRODUCING.md). The tested stack is **Python 3.10 on Windows**, with Edge and Arial for native vector export. Editable XLSX files additionally require **Node.js and `@oai/artifact-tool`**. A plain Python installation alone is not sufficient for every export; the guide records the requirements and runtime options.

From the repository root:

```powershell
# Rebuild Figure 2
python reproduce.py --figure 2 --output-dir outputs/figure2

# Recalculate the statistical results without rebuilding artwork
python reproduce.py --statistics-only --output-dir outputs/statistics

# Rebuild all figures and tables, then run the statistical replay
python reproduce.py
```

Use `ed10` for Extended Data Figure 10, or `table2` for Supplementary Table 2. Every figure/table also has its own runnable `build.py`. No additional source-only flag is needed.

Use a new output directory for each run. If none is supplied, outputs go to a uniquely named folder under `outputs/`. Builds produce the original figure export formats or PDF/XLSX tables, plus logs and verification reports. [Full commands, dependencies and output guide](docs/REPRODUCING.md).

## Repository layout

```text
figures/
  figure_01/ … figure_05/            Main figures
  extended_data_01/ … _10/          Extended Data figures
    README.md                       Purpose, data links and build command
    build.py                        Runnable figure entry point
    data/                           Derived observations, summaries and examples
    code/                           Figure-specific code and shared-helper links
tables/table_01/ … table_02/         The same structure for supplementary tables
docs/                               Reproduction, data and statistical guides
reproduction/                       Shared verified engine and technical provenance
```

## Find a figure or table

### Main figures

| Figure | Topic |
|---|---|
| [Figure 1](figures/figure_01/README.md) | Behavior across maze experience |
| [Figure 2](figures/figure_02/README.md) | Spatial representations across days |
| [Figure 3](figures/figure_03/README.md) | Within-session changes in spatial maps |
| [Figure 4](figures/figure_04/README.md) | Reward decoding and navigation |
| [Figure 5](figures/figure_05/README.md) | Maze-level representations and exploration |

### Extended Data

| Figure | Topic |
|---|---|
| [ED1](figures/extended_data_01/README.md) | Behavioral measures and speed |
| [ED2](figures/extended_data_02/README.md) | Turning policies and simulation controls |
| [ED3](figures/extended_data_03/README.md) | Navigation after latent experience |
| [ED4](figures/extended_data_04/README.md) | Neuron registration across sessions |
| [ED5](figures/extended_data_05/README.md) | Day-combination and sampling controls |
| [ED6](figures/extended_data_06/README.md) | Window-level map changes |
| [ED7](figures/extended_data_07/README.md) | Age and experience |
| [ED8](figures/extended_data_08/README.md) | Reward-region and route controls |
| [ED9](figures/extended_data_09/README.md) | Level maps and decoder controls |
| [ED10](figures/extended_data_10/README.md) | Neural–behavioral relationships |

### Supplementary tables

- [Table 1: analysis and animal membership](tables/table_01/README.md)
- [Table 2: animals, sessions and return intervals](tables/table_02/README.md)

## Scope, verification and future releases

This is the **2026-09-30 derived Source Data snapshot**. It does **not** contain the complete raw imaging/video recordings or rerun tracking, neural extraction, neuron registration, map fitting or decoder fitting. Saved null/simulator draws are replay inputs, not a claim that their entire upstream generation is repeated here. Finished composite figures are not used as rendering inputs.

All 17 artifacts in the original snapshot were rebuilt from source-only stages and compared with the approved outputs; all 53 recorded statistical groups passed replay. The source contents are preserved during the reader-facing reorganization. [Original reproduction evidence](docs/validation/REPRODUCTION_REPORT.json) · [Statistical evidence](docs/validation/STATISTICS_REPORT.json) · [Reorganization checks](docs/validation/READER_CLEANUP_REPORT.json).

After paper acceptance, the repository's main contents are planned to be replaced by the complete workflow starting from raw data. The current snapshot remains identifiable through Git history and the `source-data-2026-09-30` tag. It should not be treated as validation of that future raw-data workflow.

No new code/data licence, DOI or manuscript citation is assigned by preparing this repository. Those publication metadata remain to be supplied by the author; existing file notices are retained. [Provenance and licensing status](docs/PROVENANCE.md).
