# Supplementary Table 1: data

`current_rows.json` is the table's derived cohort source. Each object in `rows` describes one analysis: `analysis` is its name, `panels` the manuscript panels, `mice` the included animal IDs, and `note` the exclusions or support limitations. The builder calculates displayed counts from the ID lists and exports the same content to XLSX and PDF. [Field index](index.csv).

Rows are overlapping analysis cohorts, not disjoint groups of animals; do not add their counts to estimate the total number of mice. An animal missing from one row is not necessarily excluded from the study. IDs are labels, not measurements. Evidence/authority fields describe provenance and are not files required to build the table.

| File | Format |
|---|---|
| [current_rows.json](current_rows.json) | .json |
