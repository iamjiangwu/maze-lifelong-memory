"""Promote the AF23 direct-entry reconstruction into immutable figure records.

The promotion is deliberately fail closed.  It clones the two selected parent
records into temporary sibling directories, replaces only the affected figure
exports and route-score sources, validates the reconstructed 489-3day1 choice
sequence, closes each record hash inventory, and only then switches both
current-record pointers.  The selected parent records are never edited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
from PIL import Image

import validate_current_figure_records as current_records


REPO = Path(__file__).resolve().parents[1]
RECORD_ROOT = REPO / "docs/figure_version_records"
MAIN_POINTER = RECORD_ROOT / "CURRENT_MAIN_FIGURES.md"
ED_POINTER = RECORD_ROOT / "CURRENT_EXTENDED_DATA_FIGURES.md"

MAIN_PARENT_NAME = "main_figure1_2026-08-11_round4_display_refinement"
MAIN_TARGET_NAME = "main_figure1_2026-08-11_round5_direct_entry_reconstruction"
ED_PARENT_NAME = "extended_data_fig03_2026-08-08_full_audit"
ED_TARGET_NAME = "extended_data_fig03_2026-08-11_round1_direct_entry_reconstruction"

MAIN_PARENT = RECORD_ROOT / MAIN_PARENT_NAME
MAIN_TARGET = RECORD_ROOT / MAIN_TARGET_NAME
ED_PARENT = RECORD_ROOT / ED_PARENT_NAME
ED_TARGET = RECORD_ROOT / ED_TARGET_NAME

AF23_ROOT = REPO / "docs/af23_first_route_resolution_2026-08-11"
MAIN_CANDIDATE = AF23_ROOT / "main_figure1_candidate"
ED_CANDIDATE = AF23_ROOT / "extended_data_figure3_candidate"
ANALYSIS_ROOT = REPO / "docs/turn_sequence_retention_2026-08-11_direct_entry_reconstruction"
ROUTE_TABLE_ROOT = (
    REPO / "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables"
)

EXPECTED_DEPTH_HASH = "34c932a42c2b858d0b9522e74b50b517b4b381507999e19eb9910071ad71ea96"
EXPECTED_RETURN_MEAN = 0.6904761904761904
EXPECTED_RETURN_DEPTH2 = 0.41666666666666663
EXPECTED_Q = {
    "experienced_day2_vs_naive": 0.00044171178338260825,
    "experienced_day2_vs_return": 0.0001578811745682,
    "experienced_day2_vs_latent_day2": 0.043543941295848064,
    "return_vs_naive": 0.164326986350872,
    "latent_day2_vs_naive": 0.5723311597921615,
    "return_vs_latent_day2": 0.7392802946120168,
}

MAIN_ROOT_EXPORTS = {
    "main_figure1_af23_direct_entry.svg": "figure1_version_record.svg",
    "main_figure1_af23_direct_entry.pdf": "figure1_version_record.pdf",
    "main_figure1_af23_direct_entry.html": "figure1_version_record.html",
    "main_figure1_af23_direct_entry_300dpi.png": "figure1_version_record_300dpi.png",
    "main_figure1_af23_direct_entry_600dpi.png": "figure1_version_record_600dpi.png",
    "main_figure1_af23_direct_entry_600dpi.tiff": "figure1_version_record_600dpi.tiff",
}
MAIN_PANEL_H_EXPORTS = {
    f"figure1_panel_h_first_visit_route_score{suffix}": (
        f"figure1_individual_panels/figure1_panel_h_first_visit_route_score{suffix}"
    )
    for suffix in (".svg", ".pdf", ".png", ".tiff")
}
ED_ROOT_EXPORTS = {
    "ed03_reward_path_fluency_assembly.svg": "ed03_version_record.svg",
    "ed03_reward_path_fluency_assembly.pdf": "ed03_version_record.pdf",
    "ed03_reward_path_fluency_assembly.png": "ed03_version_record_600dpi.png",
    "ed03_reward_path_fluency_assembly.tiff": "ed03_version_record_600dpi.tiff",
}

MAIN_ROUTE_TABLES = (
    "ed04_session_route_current_plus_new12.csv",
    "ed04_route_group_stats_current_plus_new12.csv",
    "ed04_route_stats_current_plus_new12.csv",
)
ED_ROUTE_TABLES = (
    "ed04_depth_small_multiples_current_plus_new12.csv",
    "ed04_route_definition_current_plus_new12.csv",
    "ed04_route_events_current_plus_new12.csv",
    "ed04_session_route_current_plus_new12.csv",
    "ed04_route_group_stats_current_plus_new12.csv",
    "ed04_route_stats_current_plus_new12.csv",
)
ANALYSIS_EVIDENCE = (
    "reward_path_first_turns.csv",
    "session_route_summary.csv",
    "entry_path_reconstruction_audit.csv",
    "entry_path_choice_reconstruction_audit.csv",
    "entry_path_frame_audit.csv",
    "run_manifest.json",
    "report.md",
    "figures/entry_path_reconstruction_489-3day1.svg",
    "figures/entry_path_reconstruction_489-3day1.pdf",
    "figures/entry_path_reconstruction_489-3day1.png",
)
FROZEN_CODE = (
    "scripts/export_turn_sequence_retention.py",
    "scripts/rebuild_af23_first_route_outputs.py",
    "scripts/render_ed02_ed04_current_panels.py",
    "scripts/render_ed03_reward_path_fluency_assembly.py",
    "scripts/preview_figure1_assembly.py",
    "scripts/render_main_figs_v3_publication.py",
    "scripts/validate_af23_direct_entry_reconstruction.py",
    "scripts/validate_current_figure_records.py",
    "scripts/promote_af23_direct_entry_figure_records.py",
    "tests/test_turn_sequence_direct_entry_reconstruction.py",
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


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")


def record_tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode("utf-8"))
    return {"files": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def inventory(root: Path, *, exclude_hash_inventory: bool = False) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in root.rglob("*")
        if path.is_file()
        and not (exclude_hash_inventory and path.name == "RECORD_HASHES.csv")
    }


def write_hashes(record: Path) -> None:
    path = record / "RECORD_HASHES.csv"
    path.unlink(missing_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for source in sorted(
            (item for item in record.rglob("*") if item.is_file() and item != path),
            key=lambda item: item.relative_to(record).as_posix().lower(),
        ):
            writer.writerow(
                (
                    source.relative_to(record).as_posix(),
                    sha256(source),
                    source.stat().st_size,
                )
            )


def verify_hashes(record: Path) -> list[str]:
    path = record / "RECORD_HASHES.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        item.relative_to(record).as_posix(): item
        for item in record.rglob("*")
        if item.is_file() and item != path
    }
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors.extend(f"missing:{name}" for name in sorted(set(expected) - set(actual)))
    for name in sorted(set(actual) & set(expected)):
        row = expected[name]
        source = actual[name]
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            errors.append(f"mismatch:{name}")
    return errors


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"Unrecognized boolean token: {value!r}")


def assert_analysis_contract() -> dict[str, Any]:
    depth_path = ROUTE_TABLE_ROOT / "ed04_depth_small_multiples_current_plus_new12.csv"
    if sha256(depth_path) != EXPECTED_DEPTH_HASH:
        raise RuntimeError(
            f"Corrected depth source hash drifted: {sha256(depth_path)} != {EXPECTED_DEPTH_HASH}"
        )

    sessions = pd.read_csv(ROUTE_TABLE_ROOT / "ed04_session_route_current_plus_new12.csv")
    events = pd.read_csv(ROUTE_TABLE_ROOT / "ed04_route_events_current_plus_new12.csv")
    groups = pd.read_csv(ROUTE_TABLE_ROOT / "ed04_route_group_stats_current_plus_new12.csv")
    statistics = pd.read_csv(ROUTE_TABLE_ROOT / "ed04_route_stats_current_plus_new12.csv")
    depth = pd.read_csv(depth_path)

    row = sessions.loc[sessions["session_id"].astype(str).eq("489-3day1")]
    if len(row) != 1:
        raise RuntimeError(f"Expected one 489-3day1 summary row, found {len(row)}")
    row = row.iloc[0]
    expected_session = {
        "route_fraction_correct": 4 / 6,
        "n_depths": 6,
        "n_decisions_observed": 4,
        "n_decisions_inferred": 2,
        "n_decisions_scored": 6,
    }
    for key, expected in expected_session.items():
        if abs(float(row[key]) - expected) > 1e-12:
            raise RuntimeError(f"489-3day1 {key}={row[key]} != {expected}")
    if as_bool(row["all6_observed"]) or not as_bool(row["all6_resolved"]):
        raise RuntimeError("489-3day1 observed/resolved completeness flags are wrong")

    selected = events.loc[events["session_id"].astype(str).eq("489-3day1")].sort_values(
        "paper_depth"
    )
    if len(selected) != 6:
        raise RuntimeError(f"Expected six resolved 489-3day1 events, found {len(selected)}")
    expected_children = [1, 3, 9, 22, 46, 94]
    expected_correct = [True, False, False, True, True, True]
    expected_origin = ["inferred", "inferred", "observed", "observed", "observed", "observed"]
    if selected["chosen_child"].astype(int).tolist() != expected_children:
        raise RuntimeError("489-3day1 resolved child sequence drifted")
    if selected["reward_correct"].map(as_bool).tolist() != expected_correct:
        raise RuntimeError("489-3day1 correctness sequence drifted")
    if selected["choice_origin"].astype(str).tolist() != expected_origin:
        raise RuntimeError("489-3day1 observed/inferred provenance drifted")
    parent1 = selected.loc[selected["parent_node"].eq(1)].iloc[0]
    if not (
        int(parent1["chosen_child"]) == 3
        and as_bool(parent1["later_recorded_parent_exit_available"])
        and int(parent1["later_recorded_chosen_child"]) == 4
        and abs(float(parent1["later_recorded_abs_frame"]) - 4102.0) < 1e-12
    ):
        raise RuntimeError("Later 1->4 exit is not retained strictly as audit provenance")

    origin_counts = events["choice_origin"].astype(str).value_counts().to_dict()
    if origin_counts != {"observed": 970, "inferred": 2}:
        raise RuntimeError(f"Combined provenance counts drifted: {origin_counts}")

    return_row = groups.loc[groups["condition"].eq("return")]
    if len(return_row) != 1 or abs(
        float(return_row.iloc[0]["mean_route_fraction_correct"]) - EXPECTED_RETURN_MEAN
    ) > 1e-12:
        raise RuntimeError("Return-group mean drifted")

    got_q = statistics.set_index("comparison")["q_bh6"].astype(float).to_dict()
    if set(got_q) != set(EXPECTED_Q):
        raise RuntimeError(f"BH6 comparison set drifted: {set(got_q)}")
    for comparison, expected in EXPECTED_Q.items():
        if abs(got_q[comparison] - expected) > 1e-12:
            raise RuntimeError(f"{comparison} q={got_q[comparison]} != {expected}")

    depth2 = depth.loc[
        depth["condition"].eq("return") & depth["paper_depth"].eq(2),
        "mean_reward_correct",
    ]
    if len(depth2) != 1 or abs(float(depth2.iloc[0]) - EXPECTED_RETURN_DEPTH2) > 1e-12:
        raise RuntimeError("Corrected Return depth-2 mean is not 0.416667")

    source_validation = json.loads((AF23_ROOT / "VALIDATION.json").read_text(encoding="utf-8"))
    if source_validation.get("status") != "PASS" or source_validation.get("passed") != 14:
        raise RuntimeError("AF23 source validation is not 14/14 PASS")

    return {
        "session_id": "489-3day1",
        "assumption": "direct home to the first valid location at node 16",
        "resolved_path": [0, 1, 3, 7, 16],
        "resolved_choices": expected_children,
        "correctness": expected_correct,
        "choice_origin": expected_origin,
        "n_correct": 4,
        "resolved_denominator": 6,
        "route_fraction_correct": 4 / 6,
        "later_observed_parent_exit": {
            "edge": "1->4",
            "abs_frame": 4102,
            "use": "audit provenance only; never substituted for the inferred first 1->3 exit",
        },
        "combined_choice_provenance": origin_counts,
        "return_group_mean": EXPECTED_RETURN_MEAN,
        "return_depth2_mean": EXPECTED_RETURN_DEPTH2,
        "bh6_q": got_q,
        "depth_source_sha256": sha256(depth_path),
    }


def vector_checks(record: Path, stem: str) -> dict[str, bool]:
    pdf = record / f"{stem}.pdf"
    svg = record / f"{stem}.svg"
    png = record / f"{stem}_600dpi.png"
    tiff = record / f"{stem}_600dpi.tiff"
    with fitz.open(pdf) as document:
        pdf_one_page = document.page_count == 1
        pdf_images = sum(len(page.get_images(full=True)) for page in document)
    svg_text = svg.read_text(encoding="utf-8")
    with Image.open(png) as image:
        png_size = image.size
    with Image.open(tiff) as image:
        tiff_size = image.size
        tiff_dpi = image.info.get("dpi", (0, 0))
        tiff_compression = str(image.info.get("compression", "")).lower()
    return {
        "pdf_one_page": pdf_one_page,
        "pdf_vector_only": pdf_images == 0,
        "svg_editable_text": "<text" in svg_text,
        "svg_no_embedded_images": "<image" not in svg_text,
        "svg_no_external_links": not bool(
            re.search(r'''(?:href|xlink:href)=["'](?!#|data:)[^"']+["']''', svg_text)
        ),
        "png_tiff_pixel_dimensions_match": png_size == tiff_size,
        "tiff_600dpi": all(abs(float(value) - 600.0) < 1.0 for value in tiff_dpi),
        "tiff_lzw": tiff_compression in {"tiff_lzw", "lzw"},
    }


def move_parent_metadata(stage: Path, names: tuple[str, ...]) -> None:
    destination = stage / "parent_record_history"
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = stage / name
        if source.is_file():
            shutil.move(str(source), str(destination / name))


def freeze_evidence(stage: Path) -> None:
    evidence = stage / "source_validation/af23_direct_entry_reconstruction"
    for relative in ANALYSIS_EVIDENCE:
        copy_checked(ANALYSIS_ROOT / relative, evidence / relative)
    copy_checked(AF23_ROOT / "VALIDATION.json", evidence / "AF23_VALIDATION.json")
    copy_checked(
        ROUTE_TABLE_ROOT / "ed04_current_render_manifest.json",
        evidence / "ED04_CURRENT_RENDER_MANIFEST.json",
    )
    for relative in FROZEN_CODE:
        copy_checked(REPO / relative, stage / "source_code" / Path(relative).name)


def update_main_manifest(stage: Path, contract: dict[str, Any]) -> None:
    path = stage / "figure1_source_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["record"] = MAIN_TARGET_NAME
    payload["supersedes"] = MAIN_PARENT_NAME
    if "display_change" in payload:
        payload["inherited_display_change"] = payload.pop("display_change")
    payload["af23_direct_entry_reconstruction"] = contract
    payload["audit_tables"].update(
        {
            "route_score_session_rows": "plot_ready/ed04_session_route_current_plus_new12.csv",
            "route_score_group_summary": "plot_ready/ed04_route_group_stats_current_plus_new12.csv",
            "route_score_clustered_bh6_contrasts": "plot_ready/ed04_route_stats_current_plus_new12.csv",
            "entry_path_reconstruction": "source_validation/af23_direct_entry_reconstruction/entry_path_reconstruction_audit.csv",
            "entry_path_choices": "source_validation/af23_direct_entry_reconstruction/entry_path_choice_reconstruction_audit.csv",
        }
    )
    source = next(item for item in payload["sources"] if item["display_label"] == "h")
    source["filters"] = (
        "metric=route_fraction_correct; earliest resolved pre-first-reward exit at each of six "
        "reward-route parents; directly observed choices and interval-censored direct-home "
        "reconstructions scored together; 489-3day1 contributes inferred 0->1 and 1->3, "
        "whereas its later observed 1->4 exit is audit-only; 162 sessions from 33 mice; "
        "four-level session OLS with CR1 mouse-clustered covariance and small-G t reference; "
        "all six pairwise contrasts under BH6"
    )
    source["source_path"] = "; ".join(
        f"plot_ready/{name}" for name in MAIN_ROUTE_TABLES
    )
    payload["outputs"].update(
        {
            "html": "figure1_version_record.html",
            "png_300dpi": "figure1_version_record_300dpi.png",
        }
    )
    payload["status"] = "accepted_immutable_version_record"
    write_json(path, payload)

    verification_path = stage / "figure1_panel_verification.csv"
    with verification_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    row = next(item for item in rows if item["panel_letter"] == "h")
    row["source"] = "; ".join(f"plot_ready/{name}" for name in MAIN_ROUTE_TABLES)
    row["filters"] = (
        "earliest resolved pre-first-reward exit at six reward-route parents; observed and "
        "direct-home inferred choices scored; 489-3day1=4/6 from two inferred plus four "
        "observed choices; later 1->4 exit retained as audit only; CR1 clustered BH6"
    )
    with verification_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_ed_manifest(stage: Path, contract: dict[str, Any]) -> None:
    path = stage / "ED03_ASSEMBLY_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["record"] = ED_TARGET_NAME
    payload["supersedes"] = ED_PARENT_NAME
    payload["af23_direct_entry_reconstruction"] = contract
    payload["sources"]["depth_small_multiples"] = {
        "path": "plot_ready/ed04_depth_small_multiples_current_plus_new12.csv",
        "sha256": sha256(stage / "plot_ready/ed04_depth_small_multiples_current_plus_new12.csv"),
    }
    payload["sources"]["route_definition"] = {
        "path": "plot_ready/ed04_route_definition_current_plus_new12.csv",
        "sha256": sha256(stage / "plot_ready/ed04_route_definition_current_plus_new12.csv"),
    }
    for key, name in (
        ("route_events", "ed04_route_events_current_plus_new12.csv"),
        ("session_route", "ed04_session_route_current_plus_new12.csv"),
        ("route_group_stats", "ed04_route_group_stats_current_plus_new12.csv"),
        ("route_pairwise_stats", "ed04_route_stats_current_plus_new12.csv"),
    ):
        payload["sources"][key] = {
            "path": f"plot_ready/{name}",
            "sha256": sha256(stage / "plot_ready" / name),
        }
    payload["outputs"] = [
        {
            "path": destination,
            "sha256": sha256(stage / destination),
            "bytes": (stage / destination).stat().st_size,
        }
        for destination in ED_ROOT_EXPORTS.values()
    ]
    new_checks = [
        {
            "check": "af23_corrected_return_depth2_mean",
            "status": "PASS",
            "detail": f"{EXPECTED_RETURN_DEPTH2:.15f}; source SHA256 {EXPECTED_DEPTH_HASH}",
        },
        {
            "check": "af23_489_resolved_denominator",
            "status": "PASS",
            "detail": "4 correct / 6 resolved; 2 inferred and 4 observed choices",
        },
        {
            "check": "af23_later_parent1_exit_is_audit_only",
            "status": "PASS",
            "detail": "later observed 1->4 at frame 4102 does not replace inferred first 1->3",
        },
    ]
    payload["validation"] = [
        item for item in payload.get("validation", [])
        if not str(item.get("check", "")).startswith("af23_")
    ] + new_checks
    payload["validation_status"] = "pass"
    write_json(path, payload)

    validation_path = stage / "ED03_VALIDATION.csv"
    with validation_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if not str(row.get("check", "")).startswith("af23_")]
    rows.extend(new_checks)
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "status", "detail"))
        writer.writeheader()
        writer.writerows(rows)


def update_ed_review_report(stage: Path) -> None:
    path = stage / "REVIEW_REPORT.md"
    text = path.read_text(encoding="utf-8")
    start = text.find("**Panels e")
    end = text.find("## Round 5", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate the superseded ED3 route-source review section")
    replacement = """**Panels e-f now use the AF23 direct-entry reconstruction.** The maintained
route source is `docs/turn_sequence_retention_2026-08-11_direct_entry_reconstruction/`.
For 489-3day1, trajectory recording begins at run node 16 (absolute frame 378).
Under the author-approved direct-home assumption, the unique home-to-entry run path
`0->1->3->7->16` reconstructs the two interval-censored pre-recording choices
`0->1` (correct) and `1->3` (incorrect). Four later parent choices are directly
observed, so the session contributes 4 correct choices among 6 resolved choices.
The later recorded `1->4` exit at frame 4102 is retained in the audit tables only;
it is not treated as the first exit from parent 1.

The combined current-plus-new12 table contains 972 scored events (970 observed and
2 inferred). Panel f therefore uses the corrected Return depth-2 mean 0.416667 from
`plot_ready/ed04_depth_small_multiples_current_plus_new12.csv` (SHA-256
`34c932a42c2b858d0b9522e74b50b517b4b381507999e19eb9910071ad71ea96`).
All four cohort supports remain unchanged: 23 naive, 74 Second-day, 56 Return and
9 Latent-day2 sessions. The previous April/May route source remains available only
inside the immutable parent record and is not a current source for panels e-f.

"""
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def build_manifest(
    *,
    stage: Path,
    figure: str,
    parent_name: str,
    target_name: str,
    parent_digest: dict[str, Any],
    contract: dict[str, Any],
    changed_files: list[str],
    unchanged_checks: dict[str, bool],
) -> None:
    payload = {
        "schema": "af23-direct-entry-figure-record-promotion-v1",
        "status": "PASS",
        "figure": figure,
        "record": target_name,
        "supersedes": parent_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_decision": contract,
        "parent_record_digest_before_promotion": parent_digest,
        "changed_or_added_files": sorted(changed_files),
        "unchanged_parent_parity": unchanged_checks,
        "root_export_sha256": {
            path.name: sha256(path)
            for path in stage.glob("*_version_record*")
            if path.is_file()
        },
    }
    write_json(stage / "AF23_DIRECT_ENTRY_RECONSTRUCTION_MANIFEST.json", payload)
    write_json(
        stage / "AUDIT_PROMOTION.json",
        {
            "schema": "af23-direct-entry-promotion-audit-v1",
            "status": "PASS",
            "record": target_name,
            "supersedes": parent_name,
            "manifest": "AF23_DIRECT_ENTRY_RECONSTRUCTION_MANIFEST.json",
            "parent_record_preserved": True,
        },
    )


def stage_main(stage: Path, contract: dict[str, Any], parent_digest: dict[str, Any]) -> None:
    shutil.copytree(MAIN_PARENT, stage, dirs_exist_ok=True)
    move_parent_metadata(
        stage,
        ("AUDIT_PROMOTION.json", "DISPLAY_CHANGE_AUDIT.json", "ROUND2_DISPLAY_REFINEMENT_MANIFEST.json"),
    )
    for obsolete in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
        (stage / obsolete).unlink(missing_ok=True)

    changed: list[str] = []
    for source_name, destination_name in MAIN_ROOT_EXPORTS.items():
        copy_checked(MAIN_CANDIDATE / source_name, stage / destination_name)
        changed.append(destination_name)
    for source_name, destination_name in MAIN_PANEL_H_EXPORTS.items():
        copy_checked(MAIN_CANDIDATE / "individual_panels" / source_name, stage / destination_name)
        changed.append(destination_name)
    copy_checked(
        MAIN_CANDIDATE / "main_figure1_af23_direct_entry_layout.json",
        stage / "source_validation/af23_direct_entry_reconstruction/figure1_layout.json",
    )
    changed.append("source_validation/af23_direct_entry_reconstruction/figure1_layout.json")
    for name in MAIN_ROUTE_TABLES:
        copy_checked(ROUTE_TABLE_ROOT / name, stage / "plot_ready" / name)
        changed.append(f"plot_ready/{name}")
    freeze_evidence(stage)
    update_main_manifest(stage, contract)

    # Only panel h changes among the eight individual panels.
    parent_panels = inventory(MAIN_PARENT / "figure1_individual_panels")
    staged_panels = inventory(stage / "figure1_individual_panels")
    unchanged_panel_names = {
        name for name in parent_panels if "panel_h_first_visit_route_score" not in name
    }
    unchanged_checks = {
        "seven_non_h_panel_files_byte_identical": all(
            staged_panels.get(name) == parent_panels[name] for name in unchanged_panel_names
        ),
        "panel_h_four_formats_replaced": sum(
            "panel_h_first_visit_route_score" in name
            and staged_panels.get(name) != parent_panels.get(name)
            for name in staged_panels
        ) == 4,
        "non_route_plot_ready_tables_byte_identical": all(
            sha256(path) == sha256(stage / "plot_ready" / path.name)
            for path in (MAIN_PARENT / "plot_ready").glob("*.csv")
            if path.name not in MAIN_ROUTE_TABLES
        ),
    }
    if not all(unchanged_checks.values()):
        raise RuntimeError(f"Main Figure 1 scope parity failed: {unchanged_checks}")

    build_manifest(
        stage=stage,
        figure="Main Figure 1 panel h",
        parent_name=MAIN_PARENT_NAME,
        target_name=MAIN_TARGET_NAME,
        parent_digest=parent_digest,
        contract=contract,
        changed_files=changed,
        unchanged_checks=unchanged_checks,
    )
    (stage / "MANIFEST.md").write_text(
        f"""# {MAIN_TARGET_NAME}

This append-only record supersedes `{MAIN_PARENT_NAME}` and implements the AF23
direct-entry reconstruction in panel h. In 489-3day1, the unique direct path from
home to the first valid location (node 16) reconstructs `0->1` and `1->3` before
recording begins. Together with four directly observed choices, the session score
is 4/6. The later recorded `1->4` exit is preserved only as audit provenance and
is not called the first exit.

Only the complete Figure 1 exports, panel h exports and its three plot-ready route
tables change. All seven other individual panels and all unrelated plot-ready
tables are byte-identical to the parent. Full provenance is frozen under
`source_validation/af23_direct_entry_reconstruction/`.
""",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        """# Rebuild this Figure 1 record

Run `export_turn_sequence_retention.py`, then `rebuild_af23_first_route_outputs.py`,
then `preview_figure1_assembly.py` and `render_main_figs_v3_publication.py` using the
record-frozen sources. Validate with `validate_af23_direct_entry_reconstruction.py`
and `validate_current_figure_records.py` before creating a successor record. Never
edit this promoted directory in place.
""",
        encoding="utf-8",
    )


def stage_ed(stage: Path, contract: dict[str, Any], parent_digest: dict[str, Any]) -> None:
    shutil.copytree(ED_PARENT, stage, dirs_exist_ok=True)
    move_parent_metadata(stage, ("AUDIT_PROMOTION.json",))
    for obsolete in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
        (stage / obsolete).unlink(missing_ok=True)

    changed: list[str] = []
    for source_name, destination_name in ED_ROOT_EXPORTS.items():
        copy_checked(ED_CANDIDATE / source_name, stage / destination_name)
        changed.append(destination_name)
    copy_checked(
        ED_CANDIDATE / "ED03_ASSEMBLY_MANIFEST.json",
        stage / "ED03_ASSEMBLY_MANIFEST.json",
    )
    copy_checked(ED_CANDIDATE / "ED03_VALIDATION.csv", stage / "ED03_VALIDATION.csv")
    for name in ED_ROUTE_TABLES:
        copy_checked(ROUTE_TABLE_ROOT / name, stage / "plot_ready" / name)
        changed.append(f"plot_ready/{name}")

    # The inherited review PowerPoint depicts the parent route source. Retain it
    # as explicitly historical evidence and use the corrected candidate render
    # as the current review image.
    historical = stage / "parent_record_history"
    historical.mkdir(parents=True, exist_ok=True)
    for name in ("ed03_review_record.pptx", "ed03_review_record_render.png"):
        source = stage / name
        if source.is_file():
            shutil.move(str(source), str(historical / name))
    copy_checked(
        ED_CANDIDATE / "ed03_reward_path_fluency_assembly.png",
        stage / "ed03_review_record_render.png",
    )
    changed.append("ed03_review_record_render.png")

    freeze_evidence(stage)
    update_ed_manifest(stage, contract)
    update_ed_review_report(stage)

    parent_plot = inventory(ED_PARENT / "plot_ready")
    stage_plot = inventory(stage / "plot_ready")
    unchanged_checks = {
        "fluency_rows_byte_identical": stage_plot.get("fluency_plot_rows.csv")
        == parent_plot.get("fluency_plot_rows.csv"),
        "fluency_statistics_byte_identical": stage_plot.get("fluency_statistics.csv")
        == parent_plot.get("fluency_statistics.csv"),
        "route_definition_byte_identical": stage_plot.get(
            "ed04_route_definition_current_plus_new12.csv"
        )
        == parent_plot.get("ed04_route_definition_current_plus_new12.csv"),
        "corrected_depth_source_hash": stage_plot.get(
            "ed04_depth_small_multiples_current_plus_new12.csv"
        )
        == EXPECTED_DEPTH_HASH,
    }
    if not all(unchanged_checks.values()):
        raise RuntimeError(f"Extended Data Figure 3 scope parity failed: {unchanged_checks}")

    build_manifest(
        stage=stage,
        figure="Extended Data Figure 3 panel f",
        parent_name=ED_PARENT_NAME,
        target_name=ED_TARGET_NAME,
        parent_digest=parent_digest,
        contract=contract,
        changed_files=changed,
        unchanged_checks=unchanged_checks,
    )
    (stage / "MANIFEST.md").write_text(
        f"""# {ED_TARGET_NAME}

This append-only record supersedes `{ED_PARENT_NAME}`. Panels a-e retain their
accepted definitions; the full page is re-exported because panel f now reads the
AF23-corrected route table. For 489-3day1, two interval-censored choices are
reconstructed on the unique direct home-to-node-16 path, yielding 4/6 resolved
choices. The later observed `1->4` exit at frame 4102 remains audit-only.

The corrected panel-f Return depth-2 mean is 0.416667 and the source-table SHA-256
is `{EXPECTED_DEPTH_HASH}`. The inherited review PowerPoint is retained only under
`parent_record_history/`; `ed03_review_record_render.png` is the corrected current
render. Provenance, source tables, renderers, validators and targeted tests are
hash-frozen in this record.
""",
        encoding="utf-8",
    )


def finalize_stage(stage: Path, figure: str, stem: str) -> dict[str, Any]:
    checks = vector_checks(stage, stem)
    generic = current_records.validate_record(stage, figure, require_hashes=False)
    checks["generic_record_semantics_without_hashes"] = generic["status"] == "pass"
    if not all(checks.values()):
        raise RuntimeError(f"Pre-hash validation failed for {figure}: {checks}; {generic}")
    validation = {
        "schema": "af23-direct-entry-record-validation-v1",
        "status": "PASS",
        "record": MAIN_TARGET_NAME if figure == "main_figure_1" else ED_TARGET_NAME,
        "figure": figure,
        "checks": checks,
        "generic_validation": generic,
    }
    write_json(stage / "RECORD_VALIDATION.json", validation)
    (stage / "PROMOTION_COMPLETE.txt").write_text(
        "PASS\nAF23 direct-entry reconstruction: 489-3day1 = 4/6 resolved choices.\n"
        "Later observed 1->4 exit is audit-only, not first.\n",
        encoding="utf-8",
    )
    write_hashes(stage)
    errors = verify_hashes(stage)
    if errors:
        raise RuntimeError(f"Hash closure failed for {figure}: {errors}")
    final_generic = current_records.validate_record(stage, figure, require_hashes=True)
    if final_generic["status"] != "pass":
        raise RuntimeError(f"Final staged validation failed for {figure}: {final_generic}")
    return {"checks": checks, "generic": final_generic, "hash_errors": errors}


def update_pointers() -> None:
    main_original = MAIN_POINTER.read_text(encoding="utf-8")
    ed_original = ED_POINTER.read_text(encoding="utf-8")

    main_lines = main_original.splitlines()
    main_replaced = False
    for index, line in enumerate(main_lines):
        if line.startswith("| Figure 1 |"):
            main_lines[index] = (
                f"| Figure 1 | `docs/figure_version_records/{MAIN_TARGET_NAME}/` | "
                "Accepted eight-panel record with the AF23 direct-entry reconstruction in h: "
                "489-3day1 contributes two interval-censored choices plus four observed choices "
                "for 4/6; the later observed 1->4 exit is audit-only. |"
            )
            main_replaced = True
            break
    if not main_replaced or MAIN_PARENT_NAME not in main_original:
        raise RuntimeError("Main Figure 1 pointer is not at the expected parent record")
    main_updated = "\n".join(main_lines) + ("\n" if main_original.endswith("\n") else "")

    ed_lines = ed_original.splitlines()
    ed_replaced = False
    for index, line in enumerate(ed_lines):
        if line.startswith("| 3 | 4 |"):
            ed_lines[index] = (
                "| 3 | 4 | Reward-path fluency and first-visit choices | "
                f"`{ED_TARGET_NAME}/` | 2026-08-11 |"
            )
            ed_replaced = True
            break
    if not ed_replaced or ED_PARENT_NAME not in ed_original:
        raise RuntimeError("Extended Data Figure 3 pointer is not at the expected parent record")
    ed_updated = "\n".join(ed_lines) + ("\n" if ed_original.endswith("\n") else "")
    note = (
        "Extended Data Figure 3's 2026-08-11 successor resolves the unobserved start of "
        "489-3day1 by the unique direct home-to-node-16 path. It scores 4/6 resolved choices; "
        "the later observed 1->4 exit is audit-only.\n\n"
    )
    marker = "All ten records passed semantic/export validation"
    position = ed_updated.find(marker)
    if position < 0:
        raise RuntimeError("Could not locate Extended Data pointer insertion marker")
    ed_updated = ed_updated[:position] + note + ed_updated[position:]

    main_temp = MAIN_POINTER.with_name(f".{MAIN_POINTER.name}.af23.tmp")
    ed_temp = ED_POINTER.with_name(f".{ED_POINTER.name}.af23.tmp")
    main_temp.write_text(main_updated, encoding="utf-8")
    ed_temp.write_text(ed_updated, encoding="utf-8")
    try:
        os.replace(main_temp, MAIN_POINTER)
        os.replace(ed_temp, ED_POINTER)
    except Exception:
        MAIN_POINTER.write_text(main_original, encoding="utf-8")
        ED_POINTER.write_text(ed_original, encoding="utf-8")
        main_temp.unlink(missing_ok=True)
        ed_temp.unlink(missing_ok=True)
        raise


def make_stage(target_name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{target_name}.stage-", dir=RECORD_ROOT))


def run() -> dict[str, Any]:
    required = (
        MAIN_PARENT,
        ED_PARENT,
        MAIN_CANDIDATE,
        ED_CANDIDATE,
        ANALYSIS_ROOT,
        ROUTE_TABLE_ROOT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing promotion inputs: " + ", ".join(missing))
    if MAIN_TARGET.exists() or ED_TARGET.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable targets: {MAIN_TARGET.exists()=}, {ED_TARGET.exists()=}"
        )
    if MAIN_PARENT_NAME not in MAIN_POINTER.read_text(encoding="utf-8"):
        raise RuntimeError("Main pointer no longer selects the expected parent")
    if ED_PARENT_NAME not in ED_POINTER.read_text(encoding="utf-8"):
        raise RuntimeError("ED pointer no longer selects the expected parent")

    preflight_global = current_records.validate_all()
    baseline_failures = {
        figure: sorted(name for name, passed in result["checks"].items() if not passed)
        for figure, result in preflight_global["records"].items()
        if result["status"] != "pass"
    }

    contract = assert_analysis_contract()
    parent_before = {
        MAIN_PARENT_NAME: record_tree_digest(MAIN_PARENT),
        ED_PARENT_NAME: record_tree_digest(ED_PARENT),
    }
    main_stage = make_stage(MAIN_TARGET_NAME)
    ed_stage = make_stage(ED_TARGET_NAME)
    promoted: list[Path] = []
    main_pointer_original = MAIN_POINTER.read_text(encoding="utf-8")
    ed_pointer_original = ED_POINTER.read_text(encoding="utf-8")
    pointers_switched = False
    try:
        stage_main(main_stage, contract, parent_before[MAIN_PARENT_NAME])
        main_validation = finalize_stage(main_stage, "main_figure_1", "figure1_version_record")
        stage_ed(ed_stage, contract, parent_before[ED_PARENT_NAME])
        ed_validation = finalize_stage(ed_stage, "extended_data_3", "ed03_version_record")

        # Both records are complete and independently valid before either pointer moves.
        os.replace(main_stage, MAIN_TARGET)
        promoted.append(MAIN_TARGET)
        os.replace(ed_stage, ED_TARGET)
        promoted.append(ED_TARGET)

        for record, figure in (
            (MAIN_TARGET, "main_figure_1"),
            (ED_TARGET, "extended_data_3"),
        ):
            result = current_records.validate_record(record, figure, require_hashes=True)
            if result["status"] != "pass":
                raise RuntimeError(f"Promoted record validation failed: {result}")

        parent_after = {
            MAIN_PARENT_NAME: record_tree_digest(MAIN_PARENT),
            ED_PARENT_NAME: record_tree_digest(ED_PARENT),
        }
        if parent_after != parent_before:
            raise RuntimeError("A selected parent record changed during promotion")

        update_pointers()
        pointers_switched = True
        global_validation = current_records.validate_all()
        new_failures = {
            figure: sorted(name for name, passed in result["checks"].items() if not passed)
            for figure, result in global_validation["records"].items()
            if result["status"] != "pass"
        }
        permitted_baseline = {
            figure: failures
            for figure, failures in baseline_failures.items()
            if figure not in {"main_figure_1", "extended_data_3"}
        }
        regressions = {
            figure: failures
            for figure, failures in new_failures.items()
            if figure in {"main_figure_1", "extended_data_3"}
            or any(name not in permitted_baseline.get(figure, []) for name in failures)
        }
        if regressions:
            raise RuntimeError(
                "Global current-record validation introduced failures: "
                f"{regressions}; baseline={baseline_failures}"
            )
        if new_failures != permitted_baseline:
            raise RuntimeError(
                "Unaffected global failure set changed during promotion: "
                f"after={new_failures}; expected={permitted_baseline}"
            )

        result = {
            "schema": "af23-direct-entry-two-record-promotion-v1",
            "status": "PASS",
            "scientific_decision": contract,
            "records": {
                "main_figure_1": {
                    "record": MAIN_TARGET.relative_to(REPO).as_posix(),
                    "validation": main_validation,
                },
                "extended_data_3": {
                    "record": ED_TARGET.relative_to(REPO).as_posix(),
                    "validation": ed_validation,
                },
            },
            "parent_record_identity": {
                name: {"before": parent_before[name], "after": parent_after[name]}
                for name in parent_before
            },
            "pointer_update": {
                "main": MAIN_POINTER.relative_to(REPO).as_posix(),
                "extended_data": ED_POINTER.relative_to(REPO).as_posix(),
                "performed_after_both_record_validations": True,
            },
            "global_current_record_validation": global_validation,
            "global_validation_baseline_failures": baseline_failures,
            "global_validation_new_failures": new_failures,
            "global_validation_no_new_failures": True,
        }
        write_json(AF23_ROOT / "FIGURE_RECORD_PROMOTION_VALIDATION.json", result)
        return result
    except Exception:
        if pointers_switched:
            MAIN_POINTER.write_text(main_pointer_original, encoding="utf-8")
            ED_POINTER.write_text(ed_pointer_original, encoding="utf-8")
        for stage in (main_stage, ed_stage):
            if stage.exists() and stage.parent.resolve() == RECORD_ROOT.resolve():
                shutil.rmtree(stage)
        # If an error occurs after an atomic directory rename but before pointer
        # selection, remove only the brand-new, unselected targets so the script
        # can be retried. Existing records are never touched.
        pointer_text = MAIN_POINTER.read_text(encoding="utf-8") + ED_POINTER.read_text(
            encoding="utf-8"
        )
        for target in promoted:
            if target.exists() and target.name not in pointer_text:
                shutil.rmtree(target)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "main_record": result["records"]["main_figure_1"]["record"],
                "ed_record": result["records"]["extended_data_3"]["record"],
                "route_score_489_3day1": "4/6",
                "later_1_to_4": "audit_only",
                "global_status": result["global_current_record_validation"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
