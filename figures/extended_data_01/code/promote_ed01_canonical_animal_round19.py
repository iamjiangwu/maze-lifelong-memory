"""Create and archive the ED1 Round 19 canonical-animal successor record."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_LEDGER = ARCHIVE / "ARCHIVE_MANIFEST.csv"
PARENT_NAME = "extended_data_fig01_2026-08-15_round18_group80_matched153"
TARGET_NAME = "extended_data_fig01_2026-08-17_round19_canonical_animal_clusters"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
ANALYSIS = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_canonical_animal_clusters_2026-08-17"
)
CAPTION = (
    REPO
    / "docs/manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)

PRIMARY_OVERLAYS = {
    "ed01_endpoint_axis_session_rows_canonical29.csv": "ed01_endpoint_axis_session_rows_current153.csv",
    "ed01_endpoint_axis_cluster_stats_canonical29_bh27.csv": "ed01_endpoint_axis_cluster_stats_current153.csv",
    "ed01_tile_time_loglog_rows_canonical29.csv": "ed02_tile_time_loglog_rows_current153.csv",
    "ed01_tile_time_loglog_stats_canonical29.csv": "ed02_tile_time_loglog_stats_current153.csv",
    "ed01_animal_alias_map.csv": "ed01_animal_alias_map_canonical29.csv",
}
ARTWORK = (
    "ed01_version_record.svg",
    "ed01_version_record.pdf",
    "ed01_version_record_300dpi.png",
    "ed01_version_record_600dpi.png",
    "ed01_version_record_600dpi.tiff",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_files(root: Path) -> list[tuple[str, Path]]:
    resolved = root.resolve()
    walk_root = Path(f"\\\\?\\{resolved}") if os.name == "nt" else resolved
    rows: list[tuple[str, Path]] = []
    for directory, _, names in os.walk(walk_root):
        for name in names:
            extended = Path(directory) / name
            normal = str(extended)
            if normal.startswith("\\\\?\\"):
                normal = normal[4:]
            relative = Path(normal).relative_to(resolved).as_posix()
            rows.append((relative, extended))
    return sorted(rows, key=lambda item: item[0].lower())


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    ledger.unlink(missing_ok=True)
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for relative, path in iter_files(record):
            if relative == ledger.name:
                continue
            writer.writerow((relative, sha256(path), path.stat().st_size))


def verify_hashes(record: Path) -> list[str]:
    ledger = record / "RECORD_HASHES.csv"
    if not ledger.is_file():
        return ["missing:RECORD_HASHES.csv"]
    with ledger.open(newline="", encoding="utf-8-sig") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {relative: path for relative, path in iter_files(record) if relative != ledger.name}
    errors = [f"unhashed:{key}" for key in sorted(set(actual) - set(expected))]
    errors.extend(f"missing:{key}" for key in sorted(set(expected) - set(actual)))
    for key in sorted(set(actual) & set(expected)):
        path = actual[key]
        row = expected[key]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{key}")
    return errors


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="raise").to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def load_analysis_validation() -> dict[str, Any]:
    path = ANALYSIS / "VALIDATION.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run python scripts/analyze_ed01_canonical_animal_clusters.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or not all(payload.get("checks", {}).values()):
        raise RuntimeError("Canonical-animal analysis validation is not PASS")
    for source_name in PRIMARY_OVERLAYS:
        source = ANALYSIS / source_name
        expected = payload.get("output_hashes", {}).get(source_name)
        if not source.is_file() or expected != sha256(source):
            raise RuntimeError(f"Analysis output hash mismatch: {source_name}")
    return payload


def update_source_audit(stage: Path) -> None:
    path = stage / "SOURCE_DATA_AUDIT.csv"
    frame = pd.read_csv(path)
    roles = {
        "f-rows": "ed01_endpoint_axis_session_rows_current153.csv",
        "f-stats": "ed01_endpoint_axis_cluster_stats_current153.csv",
        "g-rows": "ed02_tile_time_loglog_rows_current153.csv",
        "g-stats": "ed02_tile_time_loglog_stats_current153.csv",
    }
    for role, name in roles.items():
        mask = frame["panel"].eq(role)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Expected one SOURCE_DATA_AUDIT row for {role}")
        local = stage / "plot_ready" / name
        frame.loc[mask, "source_path"] = f"docs/figure_version_records/{TARGET_NAME}/plot_ready/{name}"
        frame.loc[mask, "sha256"] = sha256(local)
        frame.loc[mask, "rows"] = len(pd.read_csv(local))
        frame.loc[mask, "columns"] = len(pd.read_csv(local, nrows=1).columns)
        frame.loc[mask, "promotion_status"] = "Round 19 canonical biological-animal authority"
    for index, row in frame.iterrows():
        basename = Path(str(row["source_path"])).name
        local = stage / "plot_ready" / basename
        if not local.is_file():
            continue
        frame.loc[index, "source_path"] = (
            f"docs/figure_version_records/{TARGET_NAME}/plot_ready/{basename}"
        )
        frame.loc[index, "sha256"] = sha256(local)
        frame.loc[index, "rows"] = len(pd.read_csv(local))
        frame.loc[index, "columns"] = len(pd.read_csv(local, nrows=1).columns)
    frame.to_csv(path, index=False)


def update_root_statistical_notes(stage: Path) -> None:
    round_number = 20 if "round20" in TARGET_NAME else 19
    notes_path = stage / "CAPTION_NOTES.md"
    notes = notes_path.read_text(encoding="utf-8")
    notes = notes.replace("The round-18 plots", f"The round-{round_number} plots")
    notes = notes.replace("The round-19 plots", f"The round-{round_number} plots")
    lines = notes.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("- **f:**"):
            lines[index] = (
                "- **f:** Event-acquisition comparisons use log1p outcomes, CR1 covariance "
                "clustered by 29 canonical biological animals, two-sided t(G-1), and BH "
                "correction across one 27-contrast family. Each endpoint plot retains its "
                "own y scale. All 18 displayed comparisons remain ***; the maximum displayed "
                "adjusted p value is 0.000177644 (maximum across all 27 tests, 0.000247111)."
            )
        elif line.startswith("- **g:**"):
            lines[index] = (
                "- **g:** The row-retaining model is ln(tile transitions) ~ 1 + ln(elapsed "
                "in-maze seconds), with CR1 covariance clustered by 29 canonical biological "
                "animals. Slope=0.964205, 95% CI [0.940900, 0.987511], two-sided "
                "P=2.65546e-35; n=153 sessions, df=28."
            )
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation_path = stage / "ED01_VALIDATION.csv"
    validation = pd.read_csv(validation_path, dtype=str)
    details = {
        "current153_endpoint_scope": "153 sessions / 29 canonical biological animals",
        "panel_f_bh27_family": "27 BH27 contrasts, 18 displayed on the time and tile rows; max displayed q=0.000177644; max family q=0.000247111",
        "current153_tile_time_scope": "153 session rows; stats declare 153/29",
        "tile_time_diagnostic_frozen": "log-log slope 0.964205 [0.940900, 0.987511], CR1 p=2.65546e-35, df=28",
    }
    for check, detail in details.items():
        mask = validation["check"].eq(check)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Expected one ED01_VALIDATION row for {check}")
        validation.loc[mask, "status"] = "PASS"
        validation.loc[mask, "detail"] = detail
    validation.to_csv(validation_path, index=False)

    prior = stage / "PRIOR_REVIEW_REPORT.md"
    prior_text = prior.read_text(encoding="utf-8")
    banner = (
        "# Historical review report\n\n"
        "This inherited report documents pre-Round-19 review state. Its alias-split "
        "36-cluster endpoint and tile--time statistics are superseded by "
        "`ROUND19_STATISTICAL_VALIDATION.json` and are not current authority.\n\n"
    )
    if not prior_text.startswith("# Historical review report"):
        prior.write_text(banner + prior_text, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "path_pattern": "source_validation/**/CAPTION_NOTES_ROUND*.md",
                "status": "historical_non_authoritative",
                "reason": "retained to reproduce earlier display rounds; Round 19 root notes and statistical validation supersede alias-split values",
            },
            {
                "path_pattern": "render_work/CAPTION_NOTES_ROUND*.md",
                "status": "historical_non_authoritative",
                "reason": "retained render history; not a Round 19 statistical source",
            },
            {
                "path_pattern": "source_code/assemble_ed01_supplement_v2_bundle.py",
                "status": "historical_non_authoritative",
                "reason": "frozen original assembler; Round 19 recomputation is scripts/analyze_ed01_canonical_animal_clusters.py",
            },
        ]
    ).to_csv(stage / f"ROUND{round_number}_SUPERSESSION_LEDGER.csv", index=False)


def deep_replace(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: deep_replace(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_replace(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def validate_stage(stage: Path, parent_hashes: dict[str, str]) -> dict[str, bool]:
    endpoint_rows = pd.read_csv(
        stage / "plot_ready/ed01_endpoint_axis_session_rows_current153.csv",
        dtype={"animal_id": str, "animal_id_alias": str},
    )
    endpoint_stats = pd.read_csv(stage / "plot_ready/ed01_endpoint_axis_cluster_stats_current153.csv")
    tile_rows = pd.read_csv(
        stage / "plot_ready/ed02_tile_time_loglog_rows_current153.csv",
        dtype={"animal_id": str, "animal_id_alias": str},
    )
    tile_stats = pd.read_csv(stage / "plot_ready/ed02_tile_time_loglog_stats_current153.csv")
    displayed = endpoint_stats.loc[endpoint_stats["axis_key"].isin(["maze_time_s", "maze_tiles"])]
    caption_text = CAPTION.read_text(encoding="utf-8")
    source_audit = pd.read_csv(stage / "SOURCE_DATA_AUDIT.csv")
    record_local_sources = source_audit["source_path"].astype(str).str.contains(
        "docs/figure_version_records/", regex=False
    )
    resolved_local_sources = []
    local_prefix = f"docs/figure_version_records/{TARGET_NAME}/"
    for source_path in source_audit.loc[record_local_sources, "source_path"].astype(str):
        if source_path.startswith(local_prefix):
            resolved_local_sources.append((stage / source_path[len(local_prefix) :]).is_file())
        else:
            resolved_local_sources.append((REPO / source_path).is_file())
    checks = {
        "parent_hash_closure": not verify_hashes(PARENT),
        "endpoint_153_sessions": endpoint_rows["session"].nunique() == 153,
        "endpoint_29_biological_animals": endpoint_rows["animal_id"].nunique() == 29,
        "endpoint_36_recording_aliases": endpoint_rows["animal_id_alias"].nunique() == 36,
        "tile_time_153_sessions": tile_rows["session"].nunique() == 153,
        "tile_time_29_biological_animals": tile_rows["animal_id"].nunique() == 29,
        "endpoint_family_27": len(endpoint_stats) == 27,
        "endpoint_displayed_18": len(displayed) == 18,
        "endpoint_bh27_exact": np.allclose(
            endpoint_stats["q_bh27"], bh_adjust(endpoint_stats["cluster_p_two_sided"]), rtol=0, atol=1e-15
        ),
        "endpoint_displayed_max_q_exact": np.isclose(
            displayed["q_bh27"].max(), 0.00017764394055903434, rtol=0, atol=1e-16
        ),
        "endpoint_family_max_q_exact": np.isclose(
            endpoint_stats["q_bh27"].max(), 0.0002471109894525664, rtol=0, atol=1e-16
        ),
        "tile_time_df_28": int(tile_stats.iloc[0]["df_t"]) == 28,
        "tile_time_p_exact": np.isclose(
            float(tile_stats.iloc[0]["p_cluster_t_two_sided"]),
            2.6554636157760703e-35,
            rtol=1e-12,
            atol=0,
        ),
        "caption_uses_29_animals_and_corrected_tile_p": (
            "p=2.7\\times10^{-35}$; 153 sessions from 29 animals" in caption_text
        ),
        "root_caption_notes_are_round19": (
            "29 canonical biological animals" in (stage / "CAPTION_NOTES.md").read_text(encoding="utf-8")
            and "2.65546e-35" in (stage / "CAPTION_NOTES.md").read_text(encoding="utf-8")
            and "36 mice" not in (stage / "CAPTION_NOTES.md").read_text(encoding="utf-8")
        ),
        "root_validation_is_round19": (
            "153 sessions / 29 canonical biological animals" in (stage / "ED01_VALIDATION.csv").read_text(encoding="utf-8")
            and "2.65546e-35" in (stage / "ED01_VALIDATION.csv").read_text(encoding="utf-8")
            and "153 sessions / 36 mice" not in (stage / "ED01_VALIDATION.csv").read_text(encoding="utf-8")
        ),
        "source_audit_has_no_parent_record_links": not source_audit["source_path"].astype(str).str.contains(
            PARENT_NAME, regex=False
        ).any(),
        "source_audit_record_links_resolve": bool(resolved_local_sources) and all(resolved_local_sources),
        "artwork_byte_identical_to_parent": all(
            sha256(stage / name) == parent_hashes[name] for name in ARTWORK
        ),
    }
    return {key: bool(value) for key, value in checks.items()}


def create_record() -> Path:
    if TARGET.exists():
        errors = verify_hashes(TARGET)
        if errors:
            raise RuntimeError(f"Existing target is not hash closed: {errors[:10]}")
        return TARGET
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    parent_errors = verify_hashes(PARENT)
    if parent_errors:
        raise RuntimeError(f"Parent hash closure failed: {parent_errors[:10]}")
    analysis_validation = load_analysis_validation()
    parent_hashes = {name: sha256(PARENT / name) for name in ARTWORK}

    stage = Path(tempfile.mkdtemp(prefix=".ed01-round19-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        for source_name, destination_name in PRIMARY_OVERLAYS.items():
            shutil.copy2(ANALYSIS / source_name, stage / "plot_ready" / destination_name)

        validation_dir = stage / "source_validation/round19_canonical_animal_clusters"
        validation_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "VALIDATION.json",
            "REPORT.md",
            "SOURCE_MANIFEST.csv",
            *PRIMARY_OVERLAYS.keys(),
        ):
            shutil.copy2(ANALYSIS / name, validation_dir / name)
        shutil.copy2(CAPTION, validation_dir / CAPTION.name)
        update_source_audit(stage)
        update_root_statistical_notes(stage)

        assembly_path = stage / "ED01_ASSEMBLY_MANIFEST.json"
        assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
        assembly = deep_replace(assembly, PARENT_NAME, TARGET_NAME)
        assembly["record"] = TARGET_NAME
        assembly["supersedes"] = PARENT_NAME
        assembly["round"] = 19
        assembly["generated"] = "2026-08-17"
        assembly["source_policy"].update(
            {
                "source_audit": f"docs/figure_version_records/{TARGET_NAME}/SOURCE_DATA_AUDIT.csv",
                "panel_f_inference": "153 session rows; recording aliases canonicalized to 29 biological animals; log1p OLS with CR1 covariance clustered by biological animal and Student t(G-1); BH across one 27-test family",
                "panel_g_inference": "153 session rows; log tile transitions on log elapsed in-maze time; CR1 covariance clustered by 29 biological animals and two-sided Student t(28)",
                "round19_artwork_policy": "all root artwork is byte-identical to Round 18 because session values, point estimates, fitted line, BH star categories, and geometry are unchanged",
            }
        )
        write_json(assembly_path, assembly)

        source_code = stage / "source_code"
        for name in (
            "analyze_ed01_canonical_animal_clusters.py",
            "promote_ed01_canonical_animal_round19.py",
            "promote_ed01_canonical_animal_round20.py",
            "validate_current_figure_records.py",
            "audit_manuscript_panel_cohorts.py",
            "validate_figure_caption_collection.py",
            "validate_all_figure_refinement_round2_records.py",
        ):
            source = REPO / "scripts" / name
            if source.is_file():
                shutil.copy2(source, source_code / name)

        checks = validate_stage(stage, parent_hashes)
        checks["analysis_candidate_pass"] = analysis_validation.get("status") == "PASS"
        if not all(checks.values()):
            raise RuntimeError(f"Stage validation failed: {[key for key, value in checks.items() if not value]}")
        timestamp = datetime.now(timezone.utc).isoformat()
        round_number = 20 if "round20" in TARGET_NAME else 19
        is_release_closure = round_number == 20
        promotion = {
            "schema": f"ed01-round{round_number}-canonical-animal-promotion/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "scientific_change": (
                "Release closure for the canonical 29-animal statistics: active root metadata and all record-local SOURCE_DATA_AUDIT links now resolve to the corrected authority."
                if is_release_closure
                else "Panels f and g now cluster by 29 canonical biological animals rather than 36 recording aliases."
            ),
            "endpoint_family": {
                "tests": 27,
                "displayed_tests": 18,
                "maximum_family_q_bh27": 0.0002471109894525664,
                "maximum_displayed_q_bh27": 0.00017764394055903434,
                "display_stars_changed": False,
            },
            "tile_time": {
                "n_sessions": 153,
                "n_animals": 29,
                "slope": 0.964205247524581,
                "df": 28,
                "p_two_sided": 2.6554636157760703e-35,
            },
            "artwork_sha256": parent_hashes,
            "artwork_changed": False,
            "created_utc": timestamp,
        }
        write_json(stage / f"ED01_ROUND{round_number}_PROMOTION.json", promotion)
        write_json(
            stage / f"ROUND{round_number}_STATISTICAL_VALIDATION.json",
            {
                "schema": f"ed01-round{round_number}-statistical-validation/v1",
                "status": "PASS",
                "checks": checks,
                "analysis_validation_sha256": sha256(ANALYSIS / "VALIDATION.json"),
                "caption_sha256": sha256(CAPTION),
                "validated_utc": timestamp,
            },
        )
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. Panels f and g retain "
            "their 153 session rows but canonicalize recording aliases to 29 biological "
            "animals before clustered inference. Panel f retains one 27-test "
            "Benjamini--Hochberg family; all 18 displayed significance categories are "
            "unchanged (maximum adjusted p=0.000177644). Panel g retains slope 0.964205 "
            "and now uses CR1 covariance clustered by 29 animals with t(28), yielding "
            "two-sided p=2.65546e-35. All figure artwork and geometry are byte-identical "
            "to Round 18. The archived Round 18 record preserves the alias-split tables intact.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python scripts/analyze_ed01_canonical_animal_clusters.py`, then "
            f"`python scripts/promote_ed01_canonical_animal_round{round_number}.py --create`. The "
            "figure is not rerendered because the analysis validation proves unchanged "
            "session values, point estimates, fitted line and displayed star categories.\n",
            encoding="utf-8",
        )
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": f"ed01-round{round_number}-record-validation/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "checks": checks,
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n", encoding="utf-8"
        )
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"New record hash closure failed: {errors[:10]}")
        os.replace(stage, TARGET)
        return TARGET
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = iter_files(root)
    total = 0
    for relative, path in files:
        size = path.stat().st_size
        total += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode("utf-8"))
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def archive_parent() -> dict[str, Any]:
    destination = ARCHIVE / PARENT_NAME
    if not TARGET.is_dir() or verify_hashes(TARGET):
        raise RuntimeError("A hash-closed Round 19 successor is required before archiving")
    if not PARENT.is_dir() or destination.exists():
        raise RuntimeError(
            f"Archive precondition failed: parent={PARENT.exists()} archive={destination.exists()}"
        )
    before = tree_digest(PARENT)
    with ARCHIVE_LEDGER.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if PARENT_NAME in {row["record"] for row in rows}:
        raise RuntimeError("Parent is already listed in the archive ledger")
    original_ledger = ARCHIVE_LEDGER.read_bytes()
    temporary = ARCHIVE_LEDGER.with_name(f".{ARCHIVE_LEDGER.name}.ed01-round19.tmp")
    moved = False
    try:
        os.replace(PARENT, destination)
        moved = True
        after = tree_digest(destination)
        if before != after:
            raise RuntimeError("Archived parent tree changed during move")
        rows.append(
            {
                "record": PARENT_NAME,
                "original_path": f"docs/figure_version_records/{PARENT_NAME}",
                "archive_path": f"docs/figure_version_records_archive_2026-08-09/{PARENT_NAME}",
                "file_count": str(after["files"]),
                "bytes": str(after["bytes"]),
                "tree_sha256": after["sha256"],
                "move_status": "verified",
            }
        )
        rows.sort(key=lambda row: row["record"].lower())
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, ARCHIVE_LEDGER)
        return {"status": "PASS", "archived": PARENT_NAME, "tree": after}
    except Exception:
        temporary.unlink(missing_ok=True)
        ARCHIVE_LEDGER.write_bytes(original_ledger)
        if moved and destination.exists() and not PARENT.exists():
            os.replace(destination, PARENT)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result: object = create_record().relative_to(REPO).as_posix() if args.create else archive_parent()
    print(json.dumps({"status": "PASS", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
