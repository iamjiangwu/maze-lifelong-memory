# Supplementary Table 2: data

Calendar intervals and analysis membership for the displayed recording cycles. [Column and format index](index.csv).

| File | Role and observation unit |
|---|---|
| `recording_intervals.csv` | One displayed interval: animal, nominal month, preceding/return session identifiers and dates, interval length, analysis membership and schedule note. |
| `publication_notes.json` | The table's footnotes and explicitly stated inclusion sets. |

An interval is the return Day-i calendar date minus the preceding Day-(i+1) date, in days; nominal month labels are grouping labels, not measured durations. The workbook recalculates the date differences and summaries from the 58 supplied intervals. Mouse 1616's 110-day return belongs under 6 months; mouse 618's 180-day return belongs under 12 months. The approximately three-month interval summary excludes 1616's 14-day interval and 618's 180-day interval. Inclusion in a behavioral or neural analysis is separate from inclusion in this interval summary.

An empty table cell means no interval assigned to that nominal month, not a zero-day gap. The finalized interval table contains dates directly; recording files and session-registry reconstruction are not needed.

| File | Format |
|---|---|
| [recording_intervals.csv](recording_intervals.csv) | .csv |
| [publication_notes.json](publication_notes.json) | .json |
