# ED1 reproduction code

`derived_panels.py` loads the included tables, exports statistics, and assembles the
page. `drawing.py` contains plotting and page layout only. `inference.py` recomputes
the 18 acquisition contrasts from session observations, applies BH18, and fits the
animal-clustered log-log relation between first-long-bout time and steps.

`statistics(output_dir)` can run without drawing. `render(output_dir)` writes SVG,
PDF, PNG, and statistical CSVs. No raw recording processing is performed.
