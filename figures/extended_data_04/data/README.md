# Extended Data Figure 4: data

Field-of-view displacement and neuron-matching distributions. [Column and format index](index.csv).

| Panels | Inputs and observation unit |
|---|---|
| a–b | `four_session_418_local_correlation_paper.png` and `four_session_418_coregistration_paper.png`: the displayed four-session example images. |
| c–e | `physical_fov_exact_36_source.csv`: saved cumulative animal paths, displacement from the first field of view, 36 pairwise displacements, and the plotted means and standard errors. |
| f–g | `candidate_histograms.csv`: one bin per measure and candidate class, with exact edges and directed-candidate counts. |
| h | `registered_joint_histogram.csv`: one distance × footprint-correlation bin with its registered-pair count; summing over either axis gives the displayed marginals. |
| i | `registered_score_histogram.csv`: registration-score bin edges and registered-pair counts. |

Distances and shifts ending `_um` are micrometres. `microns_per_pixel` records the image calibration: 1.66 µm/pixel for the custom miniscope and 4.0 µm/pixel for the UCLA miniscope. Footprint correlation is dimensionless; registration score measures matching evidence, not biological map stability. Directed candidates are not independent mice and can represent both query directions. Field-of-view curves summarize animals; the histograms pool technical observations. `source_summary.json` records the population counts, the score median, counts outside the displayed ranges and the example-image scale-bar calibration.

Histogram intervals include the left edge and exclude the right edge, except the last bin includes both. Distance uses 40 bins from 0 to 14 µm; footprint correlation uses 40 bins from 0 to 1; registration score uses 50 bins from 0 to 1. Zero-count bins are retained. No candidate-level records or histogram recomputation are needed to draw these panels.

The field-of-view table uses four `record_type` values:

- `cumulative_path`: the 50 displayed animal-path points, including each animal's origin; used by panels c and d.
- `displacement_summary`: five cycle means and their `sem_um`, including the origin; used by panel d.
- `transition_pair`: the 36 adjacent-session-pair displacements and their saved jittered `plot_x`; used by panel e.
- `transition_summary`: four means, standard errors (`summary_sem_percent`) and pair counts; used by panel e.

The renderer reads these coordinates and uncertainty summaries directly. It does not re-estimate field-of-view alignment or uncertainty.

| File | Format |
|---|---|
| [candidate_histograms.csv](candidate_histograms.csv) | .csv |
| [four_session_418_coregistration_paper.png](four_session_418_coregistration_paper.png) | .png |
| [four_session_418_local_correlation_paper.png](four_session_418_local_correlation_paper.png) | .png |
| [physical_fov_exact_36_source.csv](physical_fov_exact_36_source.csv) | .csv |
| [registered_joint_histogram.csv](registered_joint_histogram.csv) | .csv |
| [registered_score_histogram.csv](registered_score_histogram.csv) | .csv |
| [source_summary.json](source_summary.json) | .json |
