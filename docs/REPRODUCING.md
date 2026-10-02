# Reproduce figures, tables or statistics

Run the commands below from the repository root. The launcher checks every frozen source checksum and uses only included derived Source Data. It automatically selects source-only mode: no finished reference artwork or `-NoCompare` flag is needed.

## Software first

The verified environment is **Python 3.10 on Windows** with NumPy, pandas, SciPy, Matplotlib and the other versions recorded in [requirements.txt](../requirements.txt) and [the original environment record](../reproduction/snapshot/RUNTIME_ENVIRONMENT.json). This is a tested environment record, not a universally tested clean-install lockfile.

Figure vector exports require **Microsoft Edge and Arial**. Supplementary tables use **openpyxl** for editable XLSX files and the Python PDF renderers included here; these public dependencies are listed in `requirements.txt`.

A plain Python installation alone is not enough for every export. These are software requirements; no external scientific-data drive is needed.

After downloading or cloning the repository, open PowerShell in its root folder. With Python 3.10 installed, create an isolated environment and install the listed packages:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py --check-only
.\.venv\Scripts\python.exe reproduce.py --figure 2 --output-dir outputs/figure2
```

This does not require activating the environment or changing PowerShell's execution policy. In the commands below, use `.\.venv\Scripts\python.exe` in place of `python` when using this environment. Install Microsoft Edge and ensure Arial is available before generating figure exports. Other Python versions or operating systems have not been verified for identical artwork.

## Choose what to build

| Task | Command |
|---|---|
| Everything: 15 figures, 2 tables and statistical replay | `python reproduce.py --output-dir outputs/all` |
| Figure 2 | `python reproduce.py --figure 2 --output-dir outputs/figure2` |
| Extended Data Figure 10 | `python reproduce.py --figure ed10 --output-dir outputs/ed10` |
| Supplementary Table 2 | `python reproduce.py --figure table2 --output-dir outputs/table2` |
| All statistical groups, without artwork | `python reproduce.py --statistics-only --output-dir outputs/statistics` |
| Input integrity only | `python reproduce.py --check-only` |
| Available figure/table names | `python reproduce.py --list` |

The same build entry point is available in every figure/table folder. For example:

```powershell
python figures/figure_02/build.py --output-dir outputs/figure2
```

Use these launchers rather than running individual files in `code/` directly: the launchers arrange the bundled inputs and shared imports those modules expect.

The PowerShell alternative is `.\reproduce.ps1 -Figure 2 -OutputDir outputs/figure2`.

Use a **new or empty output directory** each time. If omitted, the launcher creates a uniquely named folder under `outputs/`. It will not write into the source-data, code or documentation folders.

## What gets produced

Figures produce their original PNG, PDF, SVG and TIFF exports. Table builders produce PDF and editable XLSX files plus their numerical ledgers. Artifact outputs are grouped by the original identifier, such as `main_figure_2/` or `supplementary_table_2/`, inside the selected output folder.

Every build has a `BUILD.log` and `REPRODUCTION_VALIDATION.json`; the enclosing folder has `REPRODUCTION_REPORT.json`. The statistical replay writes `statistics/STATISTICS_REPORT.json`, regenerated tables and comparison tables. These checks establish agreement with the saved results, not agreement with every manuscript claim; see [analysis definitions and known differences](STATISTICS.md). A single-figure build does not automatically rerun all statistics. Table summary counts/intervals are recalculated by the table builders themselves.

## Where the computation starts

This repository starts from derived session/neuron/pair/mouse observations, fitted map/decoder measurements, saved null/simulator draws and example assets. It rebuilds figures, tables and statistics, not recording extraction, tracking, neuron registration, map fitting or decoder fitting.

The public folder layout is organized for reading. The launcher restores the required input names and import paths in a temporary directory. [Technical provenance](PROVENANCE.md) records that mapping; readers do not need to navigate it to find data or run a figure.
