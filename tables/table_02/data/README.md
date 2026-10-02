# Supplementary Table 2: data

Calendar intervals and analysis membership for the displayed recording cycles. [Column and format index](index.csv).

| File | Role and observation unit |
|---|---|
| `behavior_return_points.csv` | One behavior-return session with animal ID and nominal month. |
| `calcium_long_gap_pairs.csv` | One neural source/target session pair, animal and transition. |
| `same_stage_candidate_universe_attrition.csv` | Day-1/Day-2 session linkage and recorded date-bearing session paths used to identify the preceding second day. |
| `experiment_metafile.csv` | Experiment/session identifiers used to resolve those dates. |
| `publication_notes.json` | The table's footnotes and explicitly stated inclusion sets. |

An interval is the return Day-i calendar date minus the preceding Day-(i+1) date, in days; nominal month labels are grouping labels, not measured durations. The builder derives the displayed interval ledger, cohort membership and summaries from these inputs. Mouse 1616's 110-day return belongs under 6 months; mouse 618's 180-day return belongs under 12 months. The approximately three-month interval summary excludes 1616's 14-day interval and 618's 180-day interval. Inclusion in a behavioral or neural analysis is separate from inclusion in this interval summary.

An empty table cell means no interval assigned to that nominal month, not a zero-day gap. Saved paths provide date/linkage metadata; the referenced recordings are not needed for this build.

| File | Format |
|---|---|
| [behavior_return_points.csv](behavior_return_points.csv) | .csv |
| [calcium_long_gap_pairs.csv](calcium_long_gap_pairs.csv) | .csv |
| [experiment_metafile.csv](experiment_metafile.csv) | .csv |
| [publication_notes.json](publication_notes.json) | .json |
| [same_stage_candidate_universe_attrition.csv](same_stage_candidate_universe_attrition.csv) | .csv |
