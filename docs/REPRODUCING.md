# Reproduce the figures and tables

Run these commands from the repository root. All numerical inputs are included;
no external analysis repository or finished reference figure is needed.

## Install

Use Python 3.10:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
```

On macOS or Linux, use `.venv/bin/python` instead. In the commands below,
replace `python` with that environment's executable if it is not activated.

Chrome, Chromium or Microsoft Edge is needed for browser based page exports.
The builders discover standard installations on Windows, macOS and Linux.
To select a particular browser, set both `MAZE_BROWSER` and
`MAZE_LIFELONG_MEMORY_BROWSER` to its executable path. The former serves the
behavioral renderers, the latter the neural renderers. Statistics require no
browser.

Arial gives the published typography. The neural renderers fall back to an
available sans serif font when Arial is absent. Font availability and the
FreeType version can affect text placement or rasterization without changing
numerical values. Some behavioral exports require Arial; install it before
requesting those pages.

## Run

| Task | Command |
|---|---|
| List all 18 artifacts | `python reproduce.py --list` |
| Check installation inputs without analysis | `python reproduce.py --check-only` |
| Figure 2 | `python reproduce.py --figure 2 --output-dir outputs/figure2` |
| Extended Data Figure 10 | `python reproduce.py --figure ed10 --output-dir outputs/ed10` |
| Supplementary Table 3 | `python reproduce.py --figure table3 --output-dir outputs/table3` |
| One figure's statistics | `python reproduce.py --figure ed5 --statistics-only --output-dir outputs/ed5_stats` |
| All statistical calculations | `python reproduce.py --statistics-only --output-dir outputs/statistics` |
| All figures, three tables and statistics | `python reproduce.py --output-dir outputs/all` |

Each figure or table also has a `build.py` entry point. Use a new output
directory for each run. Source Data builds do not depend on notebook order:
the required derived observations are already included.

Most statistics take seconds; exact resampling tests and large vector exports
can take several minutes. The full set of high resolution pages takes longer.
No raw imaging analysis, neural map fitting, decoder fitting or long simulation
is performed.

## Outputs and checks

Outputs are grouped by artifact, such as `main_figure_2/`. Figures produce
PDF, SVG and PNG, with TIFF where supported. Tables 1 and 2 also produce XLSX;
Table 3 produces a two page PDF from editable definitions.

`BUILD_REPORT.json` records input verification and command completion.
PASS means the requested builders completed, not that the command independently
validated their scientific methods or compared artwork pixels. Statistics
tables contain newly calculated estimates, confidence intervals and p values.
Validation evidence and reference artwork are maintained outside this package.

The updated package has been checked on Windows. The workflow includes Windows,
macOS and Linux jobs, but the previous release's cross platform results do not
certify this revision. Numerical results do not depend on installed fonts.

See [data definitions](DATA.md), [statistical methods](STATISTICS.md) and
[code entry points](CODE.md).
