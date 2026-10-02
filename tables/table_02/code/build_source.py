"""Build the comprehensive behavior-first interval ledger for Supplementary Table 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


TIMESTAMP_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})_")
RETURN_MONTHS = (3, 6, 9, 12)
EXPECTED_RETURN_COUNTS = {3: 17, 6: 17, 9: 12, 12: 10}
EXCLUDED_PAIR = "1616day1->1616bday1"
RETAINED_1616_PAIR = "1616bday1->1616b-3day1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def parse_timestamp(value: str) -> datetime:
    match = TIMESTAMP_RE.search(value)
    if not match:
        raise ValueError(f"No timestamp in {value}")
    month, day, year, hour, minute, second = map(int, match.groups())
    return datetime(2000 + year, month, day, hour, minute, second)


def registry_timestamp(value: str) -> datetime:
    paths = json.loads(value)
    if not isinstance(paths, list) or not paths:
        raise ValueError("Expected at least one registry video path")
    return min(parse_timestamp(str(path)) for path in paths)




def find_metafile_row(rows: list[dict[str, str]], session: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["animal"] == session or row["animal"].startswith(f"{session}-")
    ]
    if not matches:
        raise ValueError(f"No experiment-metafile row for {session}")
    return min(matches, key=lambda row: parse_timestamp(row["ID"]))


def find_naive_cycle(
    candidate_rows: list[dict[str, str]], animal: str
) -> dict[str, str]:
    matches = [
        row
        for row in candidate_rows
        if row["canonical_animal_id"] == animal
        and row["month_token"] == "0.0"
        and parse_bool(row["planned_stage"])
        and int(row["registry_day2_rows_found"]) >= 1
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one naive cycle for mouse {animal}; found {len(matches)}")
    return matches[0]


def find_cycle_by_day1(
    candidate_rows: list[dict[str, str]], session_day1: str
) -> dict[str, str]:
    matches = [row for row in candidate_rows if row["session_day1"] == session_day1]
    if len(matches) != 1:
        raise ValueError(f"Expected one cycle row for {session_day1}; found {len(matches)}")
    if int(matches[0]["registry_day2_rows_found"]) < 1:
        raise ValueError(f"No registry Day i+1 row for {session_day1}")
    return matches[0]


def calcium_pairs(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    pairs: dict[str, dict[str, str]] = {}
    for row in rows:
        pair = pairs.setdefault(row["pair_key"], row)
        for field in ("animal_id", "src", "dst", "transition"):
            if pair[field] != row[field]:
                raise ValueError(f"Inconsistent {field} across calcium rows for {row['pair_key']}")
    if len(pairs) != 36:
        raise ValueError(f"Expected 36 maintained calcium pairs; found {len(pairs)}")
    if EXCLUDED_PAIR in pairs:
        raise ValueError("The short mouse-1616 pair is present in the calcium source")
    return pairs


def row_from_sessions(
    *,
    animal: str,
    return_month: str,
    previous_month: str,
    cycle: dict[str, str],
    return_session: str,
    metafile_rows: list[dict[str, str]],
    behavior_roster: bool,
    calcium: bool,
    schedule_note: str,
) -> dict[str, str]:
    preceding_start = registry_timestamp(cycle["registry_day2_video_paths"])
    return_row = find_metafile_row(metafile_rows, return_session)
    return_start = parse_timestamp(return_row["ID"])
    # Recording time of day does not contribute to a calendar-day interval.
    interval = (return_start.date() - preceding_start.date()).days
    if interval <= 0:
        raise ValueError(f"Nonpositive interval for {cycle['session_day1']}->{return_session}")
    pair_key = f"{cycle['session_day1']}->{return_session}"
    return {
        "Animal ID": animal,
        "Return month": return_month,
        "Previous nominal month": previous_month,
        "Preceding Day i session": cycle["session_day1"],
        "Preceding Day i+1 session": cycle["session_day2"],
        "Preceding Day i+1 date": preceding_start.date().isoformat(),
        "Return Day i session": return_session,
        "Return Day i date": return_start.date().isoformat(),
        "Interval (d)": str(interval),
        "Fixed-month behavior return roster": str(behavior_roster),
        "Maintained calcium imaging": str(calcium),
        "Pair key": pair_key,
        "Schedule note": schedule_note,
    }


def build_rows(
    behavior_points_path: Path,
    candidate_registry_path: Path,
    experiment_metafile_path: Path,
    calcium_source_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    behavior_points = read_csv(behavior_points_path)
    candidate_rows = read_csv(candidate_registry_path)
    metafile_rows = read_csv(experiment_metafile_path)
    calcium = calcium_pairs(calcium_source_path)

    points_by_animal: dict[str, list[dict[str, str]]] = {}
    for point in behavior_points:
        points_by_animal.setdefault(point["animal_id"], []).append(point)
    for points in points_by_animal.values():
        points.sort(key=lambda row: int(row["month"]))

    rows: list[dict[str, str]] = []
    for point in behavior_points:
        return_month = int(point["month"])
        if return_month not in RETURN_MONTHS:
            continue
        animal = point["animal_id"]
        earlier_points = [
            row for row in points_by_animal[animal] if int(row["month"]) < return_month
        ]
        if earlier_points:
            previous_point = max(earlier_points, key=lambda row: int(row["month"]))
            previous_month = int(previous_point["month"])
            cycle = find_cycle_by_day1(candidate_rows, previous_point["nickname"])
        else:
            previous_month = 0
            cycle = find_naive_cycle(candidate_rows, animal)

        return_session = point["nickname"]
        pair_key = f"{cycle['session_day1']}->{return_session}"
        schedule_note = ""
        if previous_month != return_month - 3:
            schedule_note = (
                f"{return_month}-month return followed the {previous_month}-month cycle; "
                f"no {return_month - 3}-month behavior return was present"
            )
        rows.append(
            row_from_sessions(
                animal=animal,
                return_month=str(return_month),
                previous_month=str(previous_month),
                cycle=cycle,
                return_session=return_session,
                metafile_rows=metafile_rows,
                behavior_roster=True,
                calcium=pair_key in calcium,
                schedule_note=schedule_note,
            )
        )

    behavior_pair_keys = {row["Pair key"] for row in rows}
    extra_calcium = [pair for key, pair in calcium.items() if key not in behavior_pair_keys]
    if {f'{pair["src"]}->{pair["dst"]}' for pair in extra_calcium} != {RETAINED_1616_PAIR}:
        raise ValueError("Expected the retained mouse-1616 interval as the sole additional calcium pair")
    for pair in extra_calcium:
        cycle = find_cycle_by_day1(candidate_rows, pair["src"])
        rows.append(
            row_from_sessions(
                animal=pair["animal_id"],
                return_month="6",
                previous_month="Nonstandard",
                cycle=cycle,
                return_session=pair["dst"],
                metafile_rows=metafile_rows,
                behavior_roster=False,
                calcium=True,
                schedule_note=(
                    "Displayed under the 6-month return; additional calcium interval "
                    "outside the fixed-month behavior roster. The separate 14-d interval "
                    "is listed under Other intervals."
                ),
            )
        )

    # Display the short return without assigning it to a fixed-month cohort.
    rows.append(row_from_sessions(
        animal="1616", return_month="Short", previous_month="0",
        cycle=find_cycle_by_day1(candidate_rows, "1616day1"),
        return_session="1616bday1", metafile_rows=metafile_rows,
        behavior_roster=False, calcium=True,
        schedule_note="Short return. Behavior included in pooled-return ED Fig. 2b-i; pair included in Fig. 3g,j,m and ED Fig. 9o. Excluded from fixed-month and approximately 3-month summaries.",
    ))
    if rows[-1]["Interval (d)"] != "14":
        raise ValueError("Short return calendar dates no longer yield 14 days")
    rows.sort(key=lambda row: (int(row["Animal ID"]), row["Return Day i date"]))

    behavior_rows = [row for row in rows if parse_bool(row["Fixed-month behavior return roster"])]
    for month, expected in EXPECTED_RETURN_COUNTS.items():
        actual = sum(row["Return month"] == str(month) for row in behavior_rows)
        if actual != expected:
            raise ValueError(f"Return month {month}: expected {expected}, found {actual}")
    if len(behavior_rows) != 56:
        raise ValueError(f"Expected 56 behavior intervals; found {len(behavior_rows)}")
    if sum(row["Pair key"] == EXCLUDED_PAIR for row in rows) != 1:
        raise ValueError("Expected exactly one displayed short mouse-1616 pair")
    if sum(row["Pair key"] == RETAINED_1616_PAIR and row["Return month"] == "6" for row in rows) != 1:
        raise ValueError("The retained mouse-1616 interval is not assigned to the 6-month column")
    if not any(row["Animal ID"] == "418" and row["Return month"] == "12" for row in rows):
        raise ValueError("Mouse 418's 12-month behavior interval is missing")

    displayed_intervals = [float(row["Interval (d)"]) for row in rows]
    behavior_intervals = [float(row["Interval (d)"]) for row in behavior_rows]
    calcium_rows = [row for row in rows if parse_bool(row["Maintained calcium imaging"])]
    calcium_intervals = [float(row["Interval (d)"]) for row in calcium_rows]
    if len(calcium_rows) != 37:
        raise ValueError(f"Expected all 37 calcium intervals in the union; found {len(calcium_rows)}")

    summary: dict[str, object] = {
        "displayed_rows": len(rows),
        "displayed_animals": len({row["Animal ID"] for row in rows}),
        "behavior_rows": len(behavior_rows),
        "behavior_animals": len({row["Animal ID"] for row in behavior_rows}),
        "calcium_rows": len(calcium_rows),
        "calcium_animals": len({row["Animal ID"] for row in calcium_rows}),
        "behavior_and_calcium_rows": sum(
            parse_bool(row["Fixed-month behavior return roster"])
            and parse_bool(row["Maintained calcium imaging"])
            for row in rows
        ),
        "behavior_only_rows": sum(
            parse_bool(row["Fixed-month behavior return roster"])
            and not parse_bool(row["Maintained calcium imaging"])
            for row in rows
        ),
        "additional_calcium_rows": len(extra_calcium) + 1,
        "long_gap_calcium_rows": 36,
        "behavior_return_counts": EXPECTED_RETURN_COUNTS,
        "displayed_mean_days": statistics.mean(displayed_intervals),
        "displayed_sample_sd_days": statistics.stdev(displayed_intervals),
        "displayed_range_days": [min(displayed_intervals), max(displayed_intervals)],
        "behavior_mean_days": statistics.mean(behavior_intervals),
        "behavior_sample_sd_days": statistics.stdev(behavior_intervals),
        "behavior_range_days": [min(behavior_intervals), max(behavior_intervals)],
        "calcium_mean_days": statistics.mean(calcium_intervals),
        "calcium_sample_sd_days": statistics.stdev(calcium_intervals),
        "calcium_range_days": [min(calcium_intervals), max(calcium_intervals)],
        "nonstandard_rows": [
            {
                "pair_key": row["Pair key"],
                "interval_days": float(row["Interval (d)"]),
                "note": row["Schedule note"],
            }
            for row in rows
            if row["Schedule note"]
        ],
    }
    return rows, summary


def write_outputs(
    rows: list[dict[str, str]],
    summary: dict[str, object],
    output_csv: Path,
    manifest_path: Path,
    source_paths: list[Path],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "artifact": "Supplementary Table 2 comprehensive behavior-first interval ledger",
        "interval_definition": "Return Day i calendar date minus preceding Day i+1 calendar date; time of day ignored; no timezone conversion",
        **summary,
        "short_pair_excluded_from_fixed_month_and_long_gap_summaries": EXCLUDED_PAIR,
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_paths
        ],
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-points", type=Path, required=True)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--experiment-metafile", type=Path, required=True)
    parser.add_argument("--calcium-source", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = [
        args.behavior_points,
        args.candidate_registry,
        args.experiment_metafile,
        args.calcium_source,
    ]
    rows, summary = build_rows(*source_paths)
    write_outputs(rows, summary, args.output_csv, args.manifest, source_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
