"""Package and hash-close the review-only ED2 Round 11 alignment candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tmp" / "ed02_round11_alignment_2026-08-09"
VALIDATION_ROOT = REPO_ROOT / "tmp" / "ed02_round11_validator_2026-08-09"
BASE = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed02_route_policies_and_biased_walk"
)
ASSEMBLY = BASE / "assembly_round11_review"
REVIEW = BASE / "review_2026-08-09_round11"
LEDGER = BASE / "BUNDLE_HASHES_ROUND11_REVIEW.csv"
PACKAGE_MANIFEST = BASE / "ROUND11_REVIEW_PACKAGE_MANIFEST.json"
ROUND10_REVIEW = BASE / "review_2026-08-09_round10"
CURRENT_POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"

CUMULATIVE = "sup-figs-v2_ed02-slide3_round11_review.pptx"
ONE_SLIDE = "ed02-slide3_round11_review.pptx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        copy_file(path, destination / path.relative_to(source))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_transient_authority(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?i)(?:^|[\\/])tmp[\\/]", text) or re.search(r"(?i)\b[A-Z]:[\\/]", text):
        raise RuntimeError(f"Transient path in authoritative file: {path}")


def targets() -> tuple[Path, ...]:
    return ASSEMBLY, REVIEW, LEDGER, PACKAGE_MANIFEST


def safe_rollback(created: list[Path]) -> None:
    allowed = {path.resolve() for path in targets()}
    for path in reversed(created):
        if path.resolve() not in allowed or path.resolve().parent != BASE.resolve():
            raise RuntimeError(f"Unsafe rollback target: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def report_contract(path: Path, status: str, failures: int) -> dict[str, Any]:
    report = read_json(path)
    if report.get("status") != status or report.get("failure_count") != failures:
        raise RuntimeError(f"Unexpected validation report: {path}")
    return report


def preflight() -> tuple[dict[str, Any], dict[str, Any], str]:
    for path in targets():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Round 11 package target: {path}")
    panel_root = SOURCE_ROOT / "panels"
    deck_root = SOURCE_ROOT / "deck"
    for path in (
        panel_root / "RENDER_RESULT.json",
        panel_root / "SOURCE_MANIFEST.json",
        deck_root / CUMULATIVE,
        deck_root / ONE_SLIDE,
        deck_root / "round11-build-report.json",
        deck_root / "powerpoint-render-round11" / "Slide3.PNG",
        deck_root / "one-slide-powerpoint-render-round11" / "Slide1.PNG",
        VALIDATION_ROOT / "round11-validation.json",
        VALIDATION_ROOT / "round10-negative-control.json",
        CURRENT_POINTER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    positive = report_contract(VALIDATION_ROOT / "round11-validation.json", "pass", 0)
    negative = report_contract(VALIDATION_ROOT / "round10-negative-control.json", "fail", 12)
    if positive.get("check_count") != 235:
        raise RuntimeError("Round 11 validation is not the final 235-check audit")
    render = read_json(panel_root / "RENDER_RESULT.json")
    if (render.get("status"), render.get("panel_count"), render.get("panel_artifacts"), render.get("plot_ready_tables")) != ("pass", 9, 36, 16):
        raise RuntimeError("Round 11 renderer result is incomplete")
    return positive, negative, sha256(CURRENT_POINTER)


def copy_inputs() -> None:
    panel_root = SOURCE_ROOT / "panels"
    deck_root = SOURCE_ROOT / "deck"
    for directory in ("panels", "ppt_vector", "shared_assets", "plot_ready"):
        copy_tree(panel_root / directory, ASSEMBLY / directory)
    for name in (
        "CAPTION_NOTES.md",
        "PANEL_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
        "SHARED_ASSET_MANIFEST.csv",
        "SOURCE_MANIFEST.json",
        "TABLE_MANIFEST.csv",
    ):
        copy_file(panel_root / name, ASSEMBLY / name)
    render = read_json(panel_root / "RENDER_RESULT.json")
    render["output_root"] = repo_path(ASSEMBLY)
    render["packaged_status"] = "portable Round 11 review assembly; not promoted"
    write_json(ASSEMBLY / "RENDER_RESULT.json", render)
    assert_no_transient_authority(ASSEMBLY / "RENDER_RESULT.json")

    source_names = (
        "render_ed02_slide3_caption_minimal_round4.py",
        "render_ed02_slide3_layout_optimized_round5.py",
        "render_ed02_slide3_visible_alignment_round6.py",
        "render_ed02_slide3_typography_round7.py",
        "render_ed02_slide3_shared_legend_round8.py",
        "render_ed02_slide3_marker_consistent_round9.py",
        "render_ed02_slide3_compact_round10.py",
        "render_ed02_slide3_aligned_round11.py",
        "validate_ed02_round11_review.py",
        Path(__file__).name,
    )
    source_rows = []
    for name in source_names:
        source = REPO_ROOT / "scripts" / name
        destination = ASSEMBLY / "source_code" / name
        copy_file(source, destination)
        source_rows.append({"source_path": repo_path(source), "packaged_path": repo_path(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    with (ASSEMBLY / "SOURCE_CODE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    copy_file(deck_root / CUMULATIVE, REVIEW / CUMULATIVE)
    copy_file(deck_root / ONE_SLIDE, REVIEW / ONE_SLIDE)
    copy_tree(deck_root / "powerpoint-render-round11", REVIEW / "powerpoint-render-round11")
    copy_tree(deck_root / "one-slide-powerpoint-render-round11", REVIEW / "one-slide-powerpoint-render-round11")
    copy_file(deck_root / "powerpoint-render-round11" / "Slide3.PNG", REVIEW / "ed02-slide3-round11_powerpoint_render.png")
    copy_file(deck_root / "one-slide-powerpoint-render-round11" / "Slide1.PNG", REVIEW / "ed02-slide3-round11_review_192dpi.png")
    copy_file(deck_root / "cumulative-slide-03-artifact.png", REVIEW / "ed02-slide3-round11_artifact_preview.png")
    copy_file(deck_root / "cumulative-slide-03-artifact.png", REVIEW / "artifact-previews" / "cumulative-slide-03-artifact.png")
    copy_file(deck_root / "review-slide-01-artifact.png", REVIEW / "artifact-previews" / "review-slide-01-artifact.png")
    direct = {
        "frame-map-round11.json": "frame-map-round11.json",
        "source-notes-round11.txt": "authoring/source-notes-round11.txt",
        "deviation-log-round11.txt": "authoring/deviation-log-round11.txt",
        "build_ed02_slide3_round11.mjs": "authoring/historical_transcripts/build_ed02_slide3_round11.mjs",
        "render_powerpoint_round11.ps1": "authoring/historical_transcripts/render_powerpoint_round11.ps1",
        "cumulative-before-inspect.ndjson": "authoring/historical_transcripts/cumulative-before-inspect.ndjson",
        "review-before-inspect.ndjson": "authoring/historical_transcripts/review-before-inspect.ndjson",
        "cumulative-after-inspect.ndjson": "inspections/cumulative-after-inspect.ndjson",
        "review-after-inspect.ndjson": "inspections/review-after-inspect.ndjson",
        f"{CUMULATIVE}.inspect.ndjson": "inspections/cumulative-pptx-inspect.ndjson",
        f"{ONE_SLIDE}.inspect.ndjson": "inspections/review-pptx-inspect.ndjson",
        "cumulative-slide-03.layout.json": "layouts/cumulative-slide-03.layout.json",
        "review-slide-01.layout.json": "layouts/review-slide-01.layout.json",
        "round11-build-report.json": "reports/round11-build-report.json",
    }
    for source_name, destination_name in direct.items():
        copy_file(deck_root / source_name, REVIEW / destination_name)
    report_path = REVIEW / "reports" / "round11-build-report.json"
    report = read_json(report_path)
    report["cumulative"]["input"] = repo_path(ROUND10_REVIEW / "sup-figs-v2_ed02-slide3_round10_review.pptx")
    report["review"]["input"] = repo_path(ROUND10_REVIEW / "ed02-slide3_round10_review.pptx")
    report["cumulative"]["output"] = repo_path(REVIEW / CUMULATIVE)
    report["review"]["output"] = repo_path(REVIEW / ONE_SLIDE)
    report["packagedStatus"] = "portable Round 11 review metadata; review only, not promoted"
    report["authoritativePaths"] = {"assembly": repo_path(ASSEMBLY), "review": repo_path(REVIEW), "acceptedRecord": "docs/figure_version_records/extended_data_fig02_2026-08-08_round3_slide_inventory/", "templateLineage": repo_path(ROUND10_REVIEW)}
    write_json(report_path, report)
    assert_no_transient_authority(report_path)

    copy_file(VALIDATION_ROOT / "round11-validation.json", REVIEW / "reports" / "prepackage-round11-validation.json")
    copy_file(VALIDATION_ROOT / "round11-validation.txt", REVIEW / "reports" / "prepackage-round11-validation.txt")
    copy_file(VALIDATION_ROOT / "round10-negative-control.json", REVIEW / "reports" / "round10-negative-control.json")
    copy_file(VALIDATION_ROOT / "round10-negative-control.txt", REVIEW / "reports" / "round10-negative-control.txt")


def write_docs() -> None:
    (ASSEMBLY / "README.md").write_text(
        """# ED2 Round 11 axis-aligned review assembly

This review-only assembly preserves the accepted fixed-30-minute scientific evidence and the exact Round 10 slide/panel frames. Panel a's internal schematic is 0.8x with the preceding-turn label clear of the grid. The d/e/f and g/h/i x-axis baselines are aligned; d/e and g/h plot envelopes match; d/h category spacing is reduced. All sixteen plot-ready tables remain byte-identical to Round 10.

Editable SVG/PDF, outlined PowerPoint SVG, and 600-dpi PNG/TIFF exports are included. This package does not promote ED2 or change the accepted Round 3 pointer.
""",
        encoding="utf-8",
    )
    (REVIEW / "README.md").write_text(
        f"""# ED2 slide-3 Round 11 review

- `{CUMULATIVE}`: cumulative 12-slide review deck.
- `{ONE_SLIDE}`: one-slide review deck.
- `ed02-slide3-round11_powerpoint_render.png`: native PowerPoint preview.
- `ROUND11_REVIEW_VALIDATION.json`: packaged fail-closed audit.

All Round 10 frames remain unchanged. This remains review-only; the accepted Round 3 current pointer is unchanged.
""",
        encoding="utf-8",
    )
    source_notes = f"""# Stable source notes

- Accepted input record: `docs/figure_version_records/extended_data_fig02_2026-08-08_round3_slide_inventory/`
- Assembly: `{repo_path(ASSEMBLY)}/`
- Review: `{repo_path(REVIEW)}/`
- Renderer: `scripts/render_ed02_slide3_aligned_round11.py`
- Validator: `scripts/validate_ed02_round11_review.py`
- Template lineage: `{repo_path(ROUND10_REVIEW)}/`
- Scope: first 30 cumulative TimeInMaze minutes; Explore-only junction decisions; behavior-only, so neural manual masks are not applicable.
"""
    (REVIEW / "SOURCE_NOTES.md").write_text(source_notes, encoding="utf-8")
    assert_no_transient_authority(REVIEW / "SOURCE_NOTES.md")


def run_packaged_validation() -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_ed02_round11_review.py"),
        "--panel-root", str(ASSEMBLY),
        "--deck-root", str(REVIEW),
        "--build-report", str(REVIEW / "reports" / "round11-build-report.json"),
        "--cumulative-pptx", str(REVIEW / CUMULATIVE),
        "--review-pptx", str(REVIEW / ONE_SLIDE),
        "--cumulative-render", str(REVIEW / "powerpoint-render-round11" / "Slide3.PNG"),
        "--review-render", str(REVIEW / "one-slide-powerpoint-render-round11" / "Slide1.PNG"),
        "--json-out", str(REVIEW / "ROUND11_REVIEW_VALIDATION.json"),
        "--text-out", str(REVIEW / "ROUND11_REVIEW_VALIDATION.txt"),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Packaged validator failed:\n{completed.stdout}\n{completed.stderr}")
    return report_contract(REVIEW / "ROUND11_REVIEW_VALIDATION.json", "pass", 0)


def write_manifest(pointer_hash: str, positive: dict[str, Any], negative: dict[str, Any]) -> None:
    payload = {
        "schema": 1,
        "status": "pass",
        "candidate_status": "review_only_not_promoted",
        "assembly": repo_path(ASSEMBLY),
        "review": repo_path(REVIEW),
        "assembly_file_count": sum(path.is_file() for path in ASSEMBLY.rglob("*")),
        "review_file_count": sum(path.is_file() for path in REVIEW.rglob("*")),
        "validation": {"status": positive["status"], "checks": positive["check_count"], "failures": positive["failure_count"]},
        "round10_negative_control": {"status": negative["status"], "checks": negative["check_count"], "failures": negative["failure_count"]},
        "cumulative_pptx_sha256": sha256(REVIEW / CUMULATIVE),
        "one_slide_pptx_sha256": sha256(REVIEW / ONE_SLIDE),
        "native_slide3_sha256": sha256(REVIEW / "powerpoint-render-round11" / "Slide3.PNG"),
        "current_pointer_sha256_before_after": pointer_hash,
        "accepted_record_unchanged": "docs/figure_version_records/extended_data_fig02_2026-08-08_round3_slide_inventory/",
    }
    write_json(PACKAGE_MANIFEST, payload)


def write_ledger() -> int:
    rows = []
    for root in (ASSEMBLY, REVIEW):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append({"path": repo_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    rows.append({"path": repo_path(PACKAGE_MANIFEST), "bytes": PACKAGE_MANIFEST.stat().st_size, "sha256": sha256(PACKAGE_MANIFEST)})
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def check_package() -> dict[str, Any]:
    if not all(path.exists() for path in targets()):
        raise FileNotFoundError("Round 11 package is incomplete")
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8")))
    expected = {row["path"] for row in rows}
    actual = {repo_path(path) for root in (ASSEMBLY, REVIEW) for path in root.rglob("*") if path.is_file()} | {repo_path(PACKAGE_MANIFEST)}
    if len(expected) != len(rows) or expected != actual:
        raise RuntimeError({"duplicate_rows": len(expected) != len(rows), "missing": sorted(expected - actual), "extra": sorted(actual - expected)})
    for row in rows:
        path = REPO_ROOT / Path(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Hash closure failure: {path}")
    if sha256(CURRENT_POINTER) != read_json(PACKAGE_MANIFEST)["current_pointer_sha256_before_after"]:
        raise RuntimeError("Current pointer changed")
    return {"status": "pass", "ledger_rows": len(rows), "ledger_sha256": sha256(LEDGER), "manifest_sha256": sha256(PACKAGE_MANIFEST)}


def build() -> dict[str, Any]:
    positive, negative, pointer_hash = preflight()
    created: list[Path] = []
    try:
        ASSEMBLY.mkdir(parents=True)
        created.append(ASSEMBLY)
        REVIEW.mkdir(parents=True)
        created.append(REVIEW)
        copy_inputs()
        write_docs()
        positive = run_packaged_validation()
        if sha256(CURRENT_POINTER) != pointer_hash:
            raise RuntimeError("Current pointer changed during packaging")
        write_manifest(pointer_hash, positive, negative)
        created.append(PACKAGE_MANIFEST)
        rows = write_ledger()
        created.append(LEDGER)
        result = check_package()
        result.update({"assembly_files": sum(path.is_file() for path in ASSEMBLY.rglob("*")), "review_files": sum(path.is_file() for path in REVIEW.rglob("*")), "ledger_rows": rows})
        return result
    except Exception:
        safe_rollback(created)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_package() if args.check else build(), indent=2))


if __name__ == "__main__":
    main()
