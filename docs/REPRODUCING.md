# Reproduce figures, tables or statistics

Run the commands below from the repository root. The launcher checks the source checksums and uses only the included derived Source Data; finished reference artwork is not required.

## Software first

Use **Python 3.10** with [requirements-lock.txt](../requirements-lock.txt), which pins the Python packages. [requirements.txt](../requirements.txt) lists direct dependencies; the [environment record](../reproduction/snapshot/RUNTIME_ENVIRONMENT.json) describes export requirements. Browser discovery supports Windows, macOS and Linux; end-to-end checks of the current portable build are pending.

Figure vector exports require an installed **Google Chrome, Chromium or Microsoft Edge**. The shared browser helper checks `PATH` and standard application locations, or uses an explicit `MAZE_BROWSER` executable path. The scripts do not download a browser. The manuscript artwork uses Arial; font substitution can change text geometry. Supplementary tables use **openpyxl** for editable XLSX files and the included Python PDF renderers.

A plain Python installation alone is not enough for every export. These are software requirements; no external scientific-data drive is needed.

After downloading or cloning the repository, open a terminal in its root folder. On Windows with Python 3.10 installed:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py --check-only
.\.venv\Scripts\python.exe reproduce.py --figure 2 --output-dir outputs/figure2
```

On macOS or Linux, the equivalent commands are:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py --check-only
.venv/bin/python reproduce.py --figure 2 --output-dir outputs/figure2
```

Activation is optional. In subsequent commands, substitute your environment's Python executable for `python`. If a browser is not discovered, set `MAZE_BROWSER` to its executable path; it can also select one browser consistently for all figures. For example, with Chromium on `PATH`, use `$env:MAZE_BROWSER = "chromium"` in PowerShell or `export MAZE_BROWSER=chromium` in a POSIX shell. An invalid override produces an error rather than silently choosing another browser. Statistical replay (`--statistics-only`) does not require a browser or Arial.

Small text-rasterization or layout differences can occur between Matplotlib binary distributions, even at the same version. The original exports used Matplotlib 3.7.0 with FreeType 2.12.1; the tested PyPI wheel bundles FreeType 2.6.1. The statistical results are checked numerically, independently of image pixels.

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

Every build has a `BUILD.log` and `REPRODUCTION_VALIDATION.json`; the enclosing folder has `REPRODUCTION_REPORT.json`. For artwork, **PASS means that the verified inputs were accepted and the builder completed its checks successfully**. The public build does not compare the generated artwork with a reference image or establish pixel-identical rendering.

Statistical verification is separate: `statistics/STATISTICS_REPORT.json` reports numerical comparisons between recalculated results and the supplied statistical tables. A single-figure build does not automatically rerun all statistics; use `--statistics-only` for those comparisons, or build everything. Table summary counts/intervals are recalculated by the table builders themselves. See [analysis definitions](STATISTICS.md) for what each calculation tests.

## Where the computation starts

This repository starts from derived session/neuron/pair/mouse observations, fitted map/decoder measurements, saved null/simulator draws and example assets. It rebuilds figures, tables and statistics, not recording extraction, tracking, neuron registration, map fitting or decoder fitting.

The public folder layout is organized for reading. The launcher restores the required input names and import paths in a temporary directory. [Technical provenance](PROVENANCE.md) records that mapping; readers do not need to navigate it to find data or run a figure.
