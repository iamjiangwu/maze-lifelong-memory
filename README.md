# Persistent hippocampal trace of lifelong spatial memory

Source Data and reproduction code for the manuscript by **Jiang Wu, Markus Meister and Carlos Lois**.

Use this repository to rebuild the figures, tables and reported statistics from
the included derived measurements. All inputs required for these plots and
statistics are included.

## Background

The study asks what remains of a spatial memory in hippocampal CA1 after months without experience, and how prior experience shapes representations when animals return. Mice freely navigate a binary-tree maze in darkness, with six branch decisions, 64 end nodes and a fixed water port. Two-day recording cycles recur approximately every 90 days over as much as a year.

Behavioral analyses measure first long bout time, exploration efficiency, first visit route choices and junction turning biases. Biased walk models explain the exploration improvement. Longitudinal calcium imaging measures place maps, population vectors (PVs), ensemble rates, level maps and position decoding. A level map describes tuning to depth in the branching hierarchy. Neural-behavior analyses relate preserved reward coding to first long bout time and preserved level tuning to exploration efficiency.

## Contents

The repository contains **5 main figures, 10 Extended Data figures and 3 supplementary tables**, organized as:

```text
figures/figure_01/ ... figure_05/
figures/extended_data_01/ ... extended_data_10/
tables/table_01/ ... table_03/
```

Each folder has a guide, a `build.py`, `data/` and `code/`. Data include derived neuron, session, pair and animal measurements; plotted summaries; example maps; and saved simulation/shuffle values. Formats are CSV, compressed CSV, numerical arrays and JSON. [Data guide](docs/DATA.md).

The code rebuilds artwork and tables and recalculates the recorded statistical analyses. Plotted summary curves are supplied as means, uncertainty and support counts; individual measurements are retained for scatter plots and statistical tests. For example, Figure 3's first-column curves use saved means and s.e.m., whereas its experience-trend panels retain the observations used in their fits. Each data guide maps files to panels and explains their units. [Analysis definitions and correction families](docs/STATISTICS.md).

Raw imaging movies and behavior videos are not included. Tracking, neural extraction, registration, map estimation and decoder fitting are not rerun here.

## Reproduce

Use Python 3.10 and the included package requirements. Figure export uses Chrome, Chromium or Edge, discovered on Windows, macOS or Linux; the statistics-only command needs no browser. Supplementary workbooks use openpyxl. [Setup and tested platforms](docs/REPRODUCING.md) · [Code reading map](docs/CODE.md).

From the repository root:

```powershell
python reproduce.py --figure 2 --output-dir outputs/figure2
python reproduce.py --figure ed10 --output-dir outputs/ed10
python reproduce.py --statistics-only --output-dir outputs/statistics
```

Run `python reproduce.py` for all figures, tables and statistics. Use a new output directory for each run. [Statistical analyses](docs/STATISTICS.md).

## Figures and tables

| Figure | Manuscript title |
|---|---|
| [1](figures/figure_01/README.md) | Lifelong memory of a complex navigation task |
| [2](figures/figure_02/README.md) | CA1 retains a persistent trace after long periods without maze experience |
| [3](figures/figure_03/README.md) | Experience makes spatial representations more stable |
| [4](figures/figure_04/README.md) | The reward location anchors selective place stability |
| [5](figures/figure_05/README.md) | A compact structural code predicts memory guided exploration |
| [ED1](figures/extended_data_01/README.md) | Additional reward and exploration measures for lifelong spatial memory |
| [ED2](figures/extended_data_02/README.md) | Junction turning biases and biased walk models of exploration efficiency |
| [ED3](figures/extended_data_03/README.md) | Reward path fluency and first visit route choices |
| [ED4](figures/extended_data_04/README.md) | Cross-session field of view stability and neuron matching quality |
| [ED5](figures/extended_data_05/README.md) | Examples and controls for long gap neural stability |
| [ED6](figures/extended_data_06/README.md) | Convergence within recording cycles and local stability within sessions |
| [ED7](figures/extended_data_07/README.md) | Age associations with map stability |
| [ED8](figures/extended_data_08/README.md) | Reward code stability across sampling and route controls |
| [ED9](figures/extended_data_09/README.md) | Additional analyses of the persistent level code |
| [ED10](figures/extended_data_10/README.md) | Neural preservation and later behavior |
| [Table 1](tables/table_01/README.md) | Mouse cohorts by analysis |
| [Table 2](tables/table_02/README.md) | Intervals between successive recording cycles |
| [Table 3](tables/table_03/README.md) | Glossary of behavioral and neural measures |

## License

Original code: [MIT](LICENSE). Original data and documentation: [CC BY 4.0](DATA_LICENSE.md). Existing third-party notices retain their own terms.
