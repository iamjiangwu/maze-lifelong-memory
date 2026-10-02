# Supplementary Table 1: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

Run [../build.py](../build.py) from the repository root. `build_workbook.py` reads the animal lists in [current_rows.json](../data/current_rows.json), derives cohort counts and creates the editable XLSX with openpyxl. `build_pdf.py` renders the same table with reportlab. Cohort rows overlap and must not be summed to obtain a study-wide animal count. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_pdf.py](build_pdf.py)
- [build_workbook.py](build_workbook.py)
