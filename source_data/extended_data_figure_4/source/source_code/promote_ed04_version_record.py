"""Freeze the accepted Extended Data Figure 4 build as an immutable version record.

Extended Data Figure 4 is the cross-session neuron-matching quality figure, built from
slide 5 of ``sup-figs-v2.pptx`` under the merged numbering. Its sources resolve under the
OLD number, ``extended_data_fig_05``.

Mirrors the Extended Data Figure 3 promotion: copy the accepted exports, review deck,
raw-derived source tables and rendering code into a dated folder, then re-verify byte
identity of every copy and re-assert the scientific contract from the frozen payload
rather than from the live build.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed04_registration_quality"
)
ASSEMBLY = BUILD / "assembly"
REVIEW = BUILD / "review_2026-08-06"
SOURCE = BUILD / "source_from_raw"
ALIGNMENT = SOURCE / "alignment"
RECORD = (
    REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig04_2026-08-06_round7"
)
POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"

STEM = "ed04_registration_quality_assembly"
DECK = "ed04_revision_review_round7.pptx"
DECK_RENDER = "ed04_review_round7_powerpoint.png"

PAYLOAD: tuple[tuple[Path, str], ...] = (
    (ASSEMBLY / f"{STEM}.svg", "ed04_version_record.svg"),
    (ASSEMBLY / f"{STEM}.pdf", "ed04_version_record.pdf"),
    (ASSEMBLY / f"{STEM}.png", "ed04_version_record_600dpi.png"),
    (ASSEMBLY / f"{STEM}.tiff", "ed04_version_record_600dpi.tiff"),
    (ASSEMBLY / "ED04_ASSEMBLY_MANIFEST.json", "ED04_ASSEMBLY_MANIFEST.json"),
    (ASSEMBLY / "ED04_VALIDATION.csv", "ED04_VALIDATION.csv"),
    (REVIEW / DECK, "ed04_review_record.pptx"),
    (REVIEW / DECK_RENDER, "ed04_review_record_render.png"),
    (SOURCE / "ED04_SOURCE_SUMMARY.json", "source_from_raw/ED04_SOURCE_SUMMARY.json"),
    (
        SOURCE / "ed04_candidate_features_directed_long.csv",
        "source_from_raw/ed04_candidate_features_directed_long.csv",
    ),
    (
        SOURCE / "ed04_registered_pair_features_long.csv",
        "source_from_raw/ed04_registered_pair_features_long.csv",
    ),
    (SOURCE / "ed04_pair_source_audit.csv", "source_from_raw/ed04_pair_source_audit.csv"),
    (
        ALIGNMENT / "physical_fov_exact_36_source.csv",
        "source_from_raw/physical_fov_exact_36_source.csv",
    ),
    (
        REPO_ROOT / "scripts" / "render_ed04_registration_quality_assembly.py",
        "source_code/render_ed04_registration_quality_assembly.py",
    ),
    (
        REPO_ROOT / "scripts" / "build_ed04_registration_source_from_raw.py",
        "source_code/build_ed04_registration_source_from_raw.py",
    ),
    (
        REPO_ROOT / "scripts" / "ed_supplement_common.py",
        "source_code/ed_supplement_common.py",
    ),
    (
        REPO_ROOT / "scripts" / "build_figure2_registration_quality_panels.py",
        "source_code/build_figure2_registration_quality_panels.py",
    ),
    (Path(__file__), "source_code/promote_ed04_version_record.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    missing = [str(src) for src, _ in PAYLOAD if not src.exists()]
    if missing:
        raise SystemExit(f"Refusing to promote; missing payload: {missing}")

    RECORD.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, str, int]] = []
    for source, relative in PAYLOAD:
        destination = RECORD / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        same = sha256(source) == sha256(destination)
        record(f"copy_identical_{relative.replace('/', '_')}", same, relative)
        copied.append((relative, sha256(destination), destination.stat().st_size))

    validation = list((RECORD / "ED04_VALIDATION.csv").open(encoding="utf-8"))
    rows = list(csv.DictReader(validation))
    failed = [r for r in rows if r["status"] not in ("PASS", "DEVIATION")]
    deviations = sorted(r["check"] for r in rows if r["status"] == "DEVIATION")
    record(
        "frozen_validation_no_failures",
        not failed and len(rows) >= 22,
        f"{len(rows)} checks, {len(failed)} failing, {len(deviations)} deviations",
    )
    record(
        "frozen_deviation_set",
        deviations
        == [
            "deviation_candidate_class_palette",
            "deviation_figure2_cross_check_not_applied",
            "deviation_row_spacing_opened",
        ],
        ", ".join(deviations),
    )

    manifest = json.loads((RECORD / "ED04_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    summary = manifest["source_summary"]
    panels = manifest["panels"]
    record(
        "frozen_scope",
        summary["n_session_pairs"] == 36 and summary["n_animals"] == 14,
        f"{summary['n_session_pairs']} pairs / {summary['n_animals']} animals",
    )
    record(
        "frozen_raw_totals",
        summary["n_directed_candidate_pairs"] == 119444
        and summary["n_registered_pairs_before_activity_gate"] == 21412,
        "119,444 directed candidate pairings, 21,412 registered pairs, from raw CellReg",
    )
    record("frozen_panel_count", len(panels) == 9, f"{len(panels)} panels (a-i)")
    record(
        "frozen_alignment_axes",
        panels["d"]["y_label"] == "Displacement from first FOV"
        and panels["e"]["y_label"] == "Displacement from previous FOV"
        and panels["d"]["x_label"] == panels["e"]["x_label"] == "Imaging session index",
        "d referenced to the first FOV, e to the preceding session, shared session axis",
    )
    record(
        "frozen_joint_marginals",
        bool(panels["h"].get("has_marginals")),
        "panel h carries both marginal count distributions",
    )
    record(
        "frozen_no_legacy_panel",
        not any("99_legacy" in str(p.get("path", "")) for p in panels.values()),
        "no panel resolves into 99_legacy_audit_only",
    )
    record(
        "frozen_drift_covers_all_animals",
        panels["c"]["n_animals"] == 14,
        f"{panels['c']['n_animals']} animal drift paths",
    )

    with (RECORD / "RECORD_HASHES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for relative, digest, size in sorted(copied):
            writer.writerow([relative, digest, size])

    result = {
        "figure": "Extended Data Figure 4 (merged numbering; slide 5 of sup-figs-v2.pptx)",
        "sources_resolve_under": "extended_data_fig_05 (old numbering)",
        "record": str(RECORD.relative_to(REPO_ROOT)),
        "accepted_round": 7,
        "payload_files": len(copied),
        "deviations": deviations,
        "validation": checks,
        "validation_status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
    }
    (RECORD / "RECORD_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = promote()
    failed = [c for c in result["validation"] if c["status"] != "PASS"]
    print(
        json.dumps(
            {
                "record": result["record"],
                "payload_files": result["payload_files"],
                "checks": len(result["validation"]),
                "status": result["validation_status"],
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
