# Supplementary Table 2: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

Run [../build.py](../build.py) from the repository root. `build_source.py` derives a calendar-interval ledger and analysis membership from the [included session metadata](../data/README.md). `build_workbook.py` creates the editable XLSX with openpyxl; `render_publication_table.py` creates the PDF with Matplotlib. `rebuild.py` coordinates the build and `validate_workbook.py` checks workbook values/formulas. No recording files or spreadsheet application are needed. [Environment/setup](../../../docs/REPRODUCING.md).

- [build_source.py](build_source.py)
- [build_workbook.py](build_workbook.py)
- [rebuild.py](rebuild.py)
- [render_publication_table.py](render_publication_table.py)
- [validate_workbook.py](validate_workbook.py)
