# Extended Data Figure 4: data

Field-of-view displacement and neuron-matching measurements. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–b | `four_session_418_local_correlation_paper.png` and `four_session_418_coregistration_paper.png`: the displayed four-session example images. |
| c–e | `physical_fov_exact_36_source.csv`: one alignment between successive recording-cycle fields of view. Signed x/y displacements reconstruct animal paths, displacement from the first field and displacement from the preceding field. |
| f–g | `candidate_features_directed_long.csv`: one directed query/candidate neuron comparison, with candidate class, centroid distance and footprint correlation. |
| h–i | `registered_pair_features_long.csv`: one accepted matched-neuron pair; h shows the joint distance/correlation distribution, and i shows the registration score. |

Distances and shifts ending `_um` are micrometres; `_pixels` are image pixels. `microns_per_pixel` converts between them. `spatial_footprint_correlation` is dimensionless; `cellreg_registration_score` measures matching evidence, not biological map stability. Directed candidates are not independent mice and can represent both query directions. `source_summary.json` supplies the cohort and pooling counts. Field-of-view curves summarize animals; candidate histograms pool candidate observations. Missing/unsupported correlations are excluded from the stated plotting range rather than zero-filled.

| File | Format |
|---|---|
| [candidate_features_directed_long.csv](candidate_features_directed_long.csv) | .csv |
| [four_session_418_coregistration_paper.png](four_session_418_coregistration_paper.png) | .png |
| [four_session_418_local_correlation_paper.png](four_session_418_local_correlation_paper.png) | .png |
| [physical_fov_exact_36_source.csv](physical_fov_exact_36_source.csv) | .csv |
| [registered_pair_features_long.csv](registered_pair_features_long.csv) | .csv |
| [source_summary.json](source_summary.json) | .json |
