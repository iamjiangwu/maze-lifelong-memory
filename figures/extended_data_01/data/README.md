# ED1 derived inputs

| Files | Panels and contents |
| --- | --- |
| `reward_time_curve.csv`, `long_bout_time_curve.csv` | a,b: cumulative event mean/SEM against in-maze minutes |
| `reward_step_curve.csv`, `long_bout_step_curve.csv` | c,d: cumulative event mean/SEM against steps in maze |
| `acquisition_points.csv` | e: session/animal/group identifiers and time or step values for first reward, tenth reward, and first long bout |
| `long_bout_observations.csv` | f: first-long-bout seconds and steps, with session/animal/group identifiers |
| `speed_summary.csv` | g: session mean/SEM nose speed in cm/s by behavior mode, day, and return interval |
| `coverage_visits.csv`, `coverage_cycles.csv` | h: derived endpoint-visit sequence, within-round coverage, and coverage-round efficiency for two examples |
| `time_efficiency.csv`, `step_efficiency.csv` | i,j: session-first efficiency mean/SEM against minutes or steps |

Cumulative curves carry terminal counts after recording ends and stop at each
group's 80% recording-support endpoint. Efficiency curves do not carry terminal
efficiency; displayed bins require at least five sessions and satisfy the recording
support rule. Summary support columns retain sample sizes and curve endpoints.
Acquisition values contain 918 axis-specific points (459 endpoints); the long-bout
regression contains 153 sessions from 29 animals. Tests are recomputed, not copied
from stored p-values. Internal `maze_tile_transitions` and `maze_tiles` columns
measure the displayed steps in maze.
