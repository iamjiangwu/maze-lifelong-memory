# Extended Data Figure 4: data

Field-of-view displacement and neuron-matching distributions. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–b | `four_session_418_local_correlation_paper.png` and `four_session_418_coregistration_paper.png`: the displayed four-session example images. |
| c–e | `physical_fov_exact_36_source.csv`: one alignment between successive recording-cycle fields of view. Signed x/y displacements reconstruct animal paths, displacement from the first field and displacement from the preceding field. |
| f–g | `candidate_histograms.csv`: one bin per measure and candidate class, with exact edges and directed-candidate counts. |
| h | `registered_joint_histogram.csv`: one distance × footprint-correlation bin with its registered-pair count; summing over either axis gives the displayed marginals. |
| i | `registered_score_histogram.csv`: registration-score bin edges and registered-pair counts. |

Distances and shifts ending `_um` are micrometres; `_pixels` are image pixels. `microns_per_pixel` converts between them. Footprint correlation is dimensionless; registration score measures matching evidence, not biological map stability. Directed candidates are not independent mice and can represent both query directions. Field-of-view curves summarize animals; the histograms pool technical observations. `source_summary.json` records the population counts, the score median and counts outside the displayed ranges.

Histogram intervals include the left edge and exclude the right edge, except the last bin includes both. Distance uses 40 bins from 0 to 14 µm; footprint correlation uses 40 bins from 0 to 1; registration score uses 50 bins from 0 to 1. Zero-count bins are retained. No candidate-level records or histogram recomputation are needed to draw these panels.

In the field-of-view table, `source_kind=direct` means one registration alignment; `composed` means a transform composed from the alignments listed in `component_pair_codes`/`component_pair_labels` (34 direct and two composed rows).

| File | Format |
|---|---|
| [candidate_histograms.csv](candidate_histograms.csv) | .csv |
| [four_session_418_coregistration_paper.png](four_session_418_coregistration_paper.png) | .png |
| [four_session_418_local_correlation_paper.png](four_session_418_local_correlation_paper.png) | .png |
| [physical_fov_exact_36_source.csv](physical_fov_exact_36_source.csv) | .csv |
| [registered_joint_histogram.csv](registered_joint_histogram.csv) | .csv |
| [registered_score_histogram.csv](registered_score_histogram.csv) | .csv |
| [source_summary.json](source_summary.json) | .json |
