# ED1 panel-h matched 153-session window sensitivity

## Result

The historical panel-h cache was not cohort-matched to the behavioral panels. This candidate recomputes the same body-mass speed estimator on the exact 153-session ED1 behavioral registry. At 12 months it includes 10 Day 1 and 10 Day 2 recordings, rather than 4 and 4.

## Cohort accounting

| month | day | matched sessions | expected sessions |
|---:|---|---:|---:|
| 0 | day1 | 23 | 23 |
| 0 | day2 | 18 | 18 |
| 3 | day1 | 17 | 17 |
| 3 | day2 | 17 | 17 |
| 6 | day1 | 17 | 17 |
| 6 | day2 | 17 | 17 |
| 9 | day1 | 12 | 12 |
| 9 | day2 | 12 | 12 |
| 12 | day1 | 10 | 10 |
| 12 | day2 | 10 | 10 |

The current panel-h cache contains 8 of the 20 twelve-month recordings. The matched candidate restores: 418-12day1, 451-12day1, 618-12day1, 640-12day1, 644-12day1, 646-12day1, 418-12day2, 451-12day2, 618-12day2, 640-12day2, 644-12day2, 646-12day2.

## Estimand

The statistical unit is one recording session. Continuous normalized body-center coordinates are median-smoothed within continuous trajectory runs over five frames. Only adjacent frames within the same in-maze bout contribute. Segments above 80 cm/s are excluded. Speed is total retained distance divided by eligible operational mode time. Windows use cumulative in-maze time; whole session means every observed in-maze bout and is not a fixed-duration cap. Group symbols and error bars are means +/- SEM across finite session-level estimates.

## Plot-ready results

| mode | window | day | month | mean +/- SEM (cm/s) | finite n / registry n |
|---|---|---|---:|---:|---:|
| explore | First 30 min | day1 | 0 | 3.118 +/- 0.149 | 23 / 23 |
| explore | First 30 min | day1 | 3 | 4.261 +/- 0.161 | 17 / 17 |
| explore | First 30 min | day1 | 6 | 4.609 +/- 0.226 | 17 / 17 |
| explore | First 30 min | day1 | 9 | 5.000 +/- 0.318 | 12 / 12 |
| explore | First 30 min | day1 | 12 | 4.293 +/- 0.397 | 10 / 10 |
| explore | First 30 min | day2 | 0 | 3.531 +/- 0.170 | 18 / 18 |
| explore | First 30 min | day2 | 3 | 4.349 +/- 0.250 | 17 / 17 |
| explore | First 30 min | day2 | 6 | 4.241 +/- 0.265 | 17 / 17 |
| explore | First 30 min | day2 | 9 | 4.515 +/- 0.311 | 12 / 12 |
| explore | First 30 min | day2 | 12 | 4.393 +/- 0.309 | 10 / 10 |
| explore | First 60 min | day1 | 0 | 3.082 +/- 0.156 | 23 / 23 |
| explore | First 60 min | day1 | 3 | 4.193 +/- 0.164 | 17 / 17 |
| explore | First 60 min | day1 | 6 | 4.503 +/- 0.248 | 17 / 17 |
| explore | First 60 min | day1 | 9 | 5.163 +/- 0.332 | 12 / 12 |
| explore | First 60 min | day1 | 12 | 4.028 +/- 0.438 | 10 / 10 |
| explore | First 60 min | day2 | 0 | 3.354 +/- 0.181 | 18 / 18 |
| explore | First 60 min | day2 | 3 | 4.270 +/- 0.248 | 17 / 17 |
| explore | First 60 min | day2 | 6 | 4.126 +/- 0.312 | 17 / 17 |
| explore | First 60 min | day2 | 9 | 4.134 +/- 0.329 | 12 / 12 |
| explore | First 60 min | day2 | 12 | 4.147 +/- 0.264 | 10 / 10 |
| explore | Whole session | day1 | 0 | 3.045 +/- 0.178 | 23 / 23 |
| explore | Whole session | day1 | 3 | 4.126 +/- 0.202 | 17 / 17 |
| explore | Whole session | day1 | 6 | 4.232 +/- 0.273 | 17 / 17 |
| explore | Whole session | day1 | 9 | 4.628 +/- 0.366 | 12 / 12 |
| explore | Whole session | day1 | 12 | 4.002 +/- 0.354 | 10 / 10 |
| explore | Whole session | day2 | 0 | 3.141 +/- 0.189 | 18 / 18 |
| explore | Whole session | day2 | 3 | 3.798 +/- 0.302 | 17 / 17 |
| explore | Whole session | day2 | 6 | 3.626 +/- 0.344 | 17 / 17 |
| explore | Whole session | day2 | 9 | 3.611 +/- 0.403 | 12 / 12 |
| explore | Whole session | day2 | 12 | 3.513 +/- 0.335 | 10 / 10 |
| reward | First 30 min | day1 | 0 | 3.290 +/- 0.235 | 20 / 23 |
| reward | First 30 min | day1 | 3 | 5.572 +/- 0.250 | 17 / 17 |
| reward | First 30 min | day1 | 6 | 6.943 +/- 0.393 | 17 / 17 |
| reward | First 30 min | day1 | 9 | 8.821 +/- 0.515 | 12 / 12 |
| reward | First 30 min | day1 | 12 | 7.354 +/- 0.926 | 10 / 10 |
| reward | First 30 min | day2 | 0 | 7.249 +/- 0.534 | 18 / 18 |
| reward | First 30 min | day2 | 3 | 9.815 +/- 0.420 | 17 / 17 |
| reward | First 30 min | day2 | 6 | 10.162 +/- 0.639 | 17 / 17 |
| reward | First 30 min | day2 | 9 | 11.577 +/- 0.556 | 12 / 12 |
| reward | First 30 min | day2 | 12 | 11.036 +/- 0.669 | 10 / 10 |
| reward | First 60 min | day1 | 0 | 3.687 +/- 0.254 | 23 / 23 |
| reward | First 60 min | day1 | 3 | 6.740 +/- 0.278 | 17 / 17 |
| reward | First 60 min | day1 | 6 | 8.077 +/- 0.567 | 17 / 17 |
| reward | First 60 min | day1 | 9 | 10.511 +/- 0.562 | 12 / 12 |
| reward | First 60 min | day1 | 12 | 8.470 +/- 0.965 | 10 / 10 |
| reward | First 60 min | day2 | 0 | 7.677 +/- 0.523 | 18 / 18 |
| reward | First 60 min | day2 | 3 | 10.402 +/- 0.390 | 17 / 17 |
| reward | First 60 min | day2 | 6 | 10.328 +/- 0.625 | 17 / 17 |
| reward | First 60 min | day2 | 9 | 11.675 +/- 0.561 | 12 / 12 |
| reward | First 60 min | day2 | 12 | 11.171 +/- 0.617 | 10 / 10 |
| reward | Whole session | day1 | 0 | 6.443 +/- 0.409 | 23 / 23 |
| reward | Whole session | day1 | 3 | 8.379 +/- 0.468 | 17 / 17 |
| reward | Whole session | day1 | 6 | 8.891 +/- 0.507 | 17 / 17 |
| reward | Whole session | day1 | 9 | 10.732 +/- 0.530 | 12 / 12 |
| reward | Whole session | day1 | 12 | 9.594 +/- 0.714 | 10 / 10 |
| reward | Whole session | day2 | 0 | 8.077 +/- 0.531 | 18 / 18 |
| reward | Whole session | day2 | 3 | 10.185 +/- 0.498 | 17 / 17 |
| reward | Whole session | day2 | 6 | 10.327 +/- 0.651 | 17 / 17 |
| reward | Whole session | day2 | 9 | 11.832 +/- 0.478 | 12 / 12 |
| reward | Whole session | day2 | 12 | 10.796 +/- 0.609 | 10 / 10 |

## Conversion and validation audit

- Raw and trajectory inputs loaded for all 153 sessions; cumulative in-maze duration ranged from 84.3 to 288.0 min.
- Whole-minus-first-30 group differences ranged from -0.904 to +3.153 cm/s across modes, days and months.
- `cohort_match_audit.csv` exposes every shared, omitted and extra historical panel-h session.
- `summary_inclusion_audit.csv` exposes finite-mode and reward-context inclusion masks before aggregation.
- `speed_by_5min_bin.csv` and `speed_by_session_window.csv` expose the bin-to-session transformation before group means.
- Validation passed 17/17 checks.
- This is an unpromoted candidate. The accepted ED1 record, pointers and manuscript captions were not modified.
