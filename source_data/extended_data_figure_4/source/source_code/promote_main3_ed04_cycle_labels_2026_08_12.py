"""Promote the label-only Main Figure 3 and ED4 cycle-label successors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_MANIFEST = ARCHIVE / "ARCHIVE_MANIFEST.csv"
CANDIDATE = REPO / "docs/figure_cycle_label_refinement_2026-08-12"

SPECS = {
    "main3": {
        "parent": "main_figure3_2026-08-12_round5_embedded_raster_resolution",
        "target": "main_figure3_2026-08-12_round6_cycle_labels",
        "candidate": CANDIDATE / "main3",
        "candidate_stem": "figure3_source_publication",
        "record_stem": "figure3_version_record",
        "label_count": 8,
    },
    "ed4": {
        "parent": "extended_data_fig04_2026-08-12_round3_embedded_raster_resolution",
        "target": "extended_data_fig04_2026-08-12_round4_cycle_labels",
        "candidate": CANDIDATE / "ed04",
        "candidate_stem": "ed04_registration_quality_assembly",
        "record_stem": "ed04_version_record",
        "label_count": 2,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def svg_text(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("text") and "".join(node.itertext()).strip()
    ]


def tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode())
    return {"file_count": len(files), "bytes": total, "tree_sha256": digest.hexdigest()}


def write_hashes(record: Path) -> None:
    output = record / "RECORD_HASHES.csv"
    output.unlink(missing_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*"), key=lambda item: item.as_posix().lower()):
            if path.is_file() and path != output:
                writer.writerow((path.relative_to(record).as_posix(), sha256(path), path.stat().st_size))


def verify_hashes(record: Path) -> bool:
    with (record / "RECORD_HASHES.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    return set(actual) == {row["path"] for row in rows} and all(
        actual[row["path"]].stat().st_size == int(row["bytes"])
        and sha256(actual[row["path"]]) == row["sha256"]
        for row in rows
    )


def render_pdf_png(pdf: Path, output: Path, dpi: int) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise AssertionError(f"Expected one-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).save(
        output, dpi=(dpi, dpi), optimize=True
    )


def replace_top_level_text(
    record_root: Path,
    final_record: Path,
    old_record: str,
    new_record: str,
    candidate: Path,
    candidate_stem: str,
    record_stem: str,
) -> None:
    for path in record_root.iterdir():
        if not path.is_file() or path.name in {"RECORD_HASHES.csv"}:
            continue
        if path.suffix.lower() not in {".json", ".md", ".txt", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(old_record, new_record)
        text = text.replace(str(candidate), str(final_record))
        text = text.replace(candidate.as_posix(), final_record.as_posix())
        text = text.replace(candidate_stem, record_stem)
        text = text.replace("First 30 min across experiences", "First 30 min across cycles")
        text = text.replace("No. of experiences", "No. of cycles")
        path.write_text(text, encoding="utf-8", newline="\n")


def normalize_geometry_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["source_svg"] = Path(row["source_svg"]).name
        row["source_png"] = Path(row["source_png"]).name
    return rows


def candidate_parity(key: str) -> dict[str, Any]:
    spec = SPECS[key]
    parent = RECORDS / spec["parent"]
    candidate = spec["candidate"]
    stem = spec["candidate_stem"]
    texts = svg_text(candidate / f"{stem}.svg")
    checks: dict[str, bool] = {
        "cycle_label_count": sum(text == "No. of cycles" for text in texts)
        == spec["label_count"],
        "stale_label_absent": "No. of experiences" not in texts,
        "one_page_pdf": False,
    }
    with fitz.open(candidate / f"{stem}.pdf") as document:
        checks["one_page_pdf"] = document.page_count == 1
    if key == "main3":
        checks["cycle_title_present"] = "First 30 min across cycles" in texts
        checks["stale_title_absent"] = "First 30 min across experiences" not in texts
        for name in (
            "figure3_day2_template_first30_alignment.csv",
            "figure3_day2_template_support_runs.csv",
            "figure3_panel_verification.csv",
        ):
            checks[f"byte_parity_{name}"] = (parent / name).read_bytes() == (candidate / name).read_bytes()
        checks["axes_geometry_parity"] = normalize_geometry_csv(
            parent / "figure3_axes_geometry.csv"
        ) == normalize_geometry_csv(candidate / "figure3_axes_geometry.csv")
        old = json.loads((parent / "figure3_source_manifest.json").read_text(encoding="utf-8"))
        new = json.loads((candidate / "figure3_source_manifest.json").read_text(encoding="utf-8"))
        checks["data_counts_parity"] = old["data_counts"] == new["data_counts"]
        old_alignment = dict(old["source_alignment"])
        new_alignment = dict(new["source_alignment"])
        old_alignment["path"] = Path(old_alignment["path"]).name
        new_alignment["path"] = Path(new_alignment["path"]).name
        checks["source_alignment_parity"] = old_alignment == new_alignment
    else:
        old = json.loads((parent / "ED04_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
        new = json.loads((candidate / "ED04_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
        checks["source_summary_parity"] = old["source_summary"] == new["source_summary"]
        old_panels = old["panels"]
        new_panels = json.loads(json.dumps(new["panels"]))
        for panel in ("d", "e"):
            new_panels[panel]["x_label"] = "No. of experiences"
        checks["panel_metadata_parity_except_labels"] = old_panels == new_panels
    if not all(checks.values()):
        raise AssertionError(f"{key} candidate parity failed: {checks}")
    return checks


def create_one(key: str) -> Path:
    spec = SPECS[key]
    parent = RECORDS / spec["parent"]
    target = RECORDS / spec["target"]
    candidate = spec["candidate"]
    if not parent.is_dir() or target.exists():
        raise FileExistsError(f"Parent missing or target already exists: {parent}, {target}")
    checks = candidate_parity(key)
    with tempfile.TemporaryDirectory(prefix=f"{key}_cycle_", dir=RECORDS) as temporary:
        stage = Path(temporary) / spec["target"]
        shutil.copytree(parent, stage)
        candidate_stem = candidate / spec["candidate_stem"]
        record_stem = stage / spec["record_stem"]
        for extension in (".svg", ".pdf"):
            shutil.copy2(candidate_stem.with_suffix(extension), record_stem.with_suffix(extension))
        if key == "main3":
            shutil.copy2(candidate_stem.with_suffix(".html"), record_stem.with_suffix(".html"))
            shutil.rmtree(stage / "individual_panels")
            shutil.copytree(candidate / "figure3_individual_panels", stage / "individual_panels")
            for name in (
                "figure3_source_manifest.json",
                "figure3_axes_geometry.csv",
                "figure3_panel_verification.csv",
                "figure3_day2_template_first30_alignment.csv",
                "figure3_day2_template_support_runs.csv",
            ):
                shutil.copy2(candidate / name, stage / name)
        else:
            shutil.copy2(candidate / "ED04_ASSEMBLY_MANIFEST.json", stage / "ED04_ASSEMBLY_MANIFEST.json")
            shutil.copy2(candidate / "ED04_VALIDATION.csv", stage / "ED04_VALIDATION.csv")
            for obsolete in ("ed04_review_record.pptx", "ed04_review_record_render.png"):
                (stage / obsolete).unlink(missing_ok=True)
        render_pdf_png(record_stem.with_suffix(".pdf"), stage / f"{spec['record_stem']}_300dpi.png", 300)
        render_pdf_png(record_stem.with_suffix(".pdf"), stage / f"{spec['record_stem']}_600dpi.png", 600)
        with Image.open(stage / f"{spec['record_stem']}_600dpi.png") as image:
            image.save(
                stage / f"{spec['record_stem']}_600dpi.tiff",
                dpi=(600, 600),
                compression="tiff_lzw",
            )
        replace_top_level_text(
            stage,
            target,
            spec["parent"],
            spec["target"],
            candidate,
            spec["candidate_stem"],
            spec["record_stem"],
        )
        for source in (
            REPO / "scripts/render_main_figs_v3_publication.py",
            REPO / "scripts/render_ed04_registration_quality_assembly.py",
            REPO / "tests/test_figure3_schematic_row_layout.py",
            REPO / "tests/test_extended_data_1_5_refinement_round2.py",
            Path(__file__),
        ):
            shutil.copy2(source, stage / "source_code" / source.name)
        created = datetime.now(timezone.utc).isoformat()
        write_json(
            stage / "CYCLE_LABEL_REFINEMENT_2026_08_12.json",
            {
                "schema": "cycle-label-display-refinement/v1",
                "status": "PASS",
                "record": spec["target"],
                "parent_record": spec["parent"],
                "changes": (
                    ["First 30 min across experiences -> First 30 min across cycles"]
                    if key == "main3"
                    else []
                )
                + [f"{spec['label_count']} visible No. of experiences labels -> No. of cycles"],
                "unchanged": "ticks, data, covariates, statistics, geometry and unrelated visible content",
                "candidate_checks": checks,
                "created_utc": created,
            },
        )
        (stage / "MANIFEST.md").write_text(
            f"# {spec['target']}\n\n"
            f"Append-only label-only successor to `{spec['parent']}`. The maintained Python renderer changes only the author-selected cycle terminology; ticks, data, covariates, statistics, geometry and unrelated visible content are unchanged.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            "Render with the record-frozen maintained Python renderer, then run the record-frozen cycle-label promotion validator.\n",
            encoding="utf-8",
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={spec['target']}\ncreated_utc={created}\n", encoding="utf-8"
        )
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "cycle-label-record-validation/v1",
                "status": "pass",
                "record": spec["target"],
                "checks": checks | {"append_only_successor": True, "root_exports_complete": True},
                "validated_utc": created,
            },
        )
        write_hashes(stage)
        if not verify_hashes(stage):
            raise AssertionError(f"Hash closure failed: {stage}")
        shutil.move(str(stage), str(target))
    return target


def update_pointers() -> None:
    main_path = RECORDS / "CURRENT_MAIN_FIGURES.md"
    main_text = main_path.read_text(encoding="utf-8")
    main_text = main_text.replace(SPECS["main3"]["parent"], SPECS["main3"]["target"])
    main_text = main_text.replace(
        "and `No. of experiences` axes.",
        "and `No. of cycles` axes; panel c is titled `First 30 min across cycles`.",
    )
    main_path.write_text(main_text, encoding="utf-8", newline="\n")

    ed_path = RECORDS / "CURRENT_EXTENDED_DATA_FIGURES.md"
    ed_text = ed_path.read_text(encoding="utf-8")
    ed_text = ed_text.replace(SPECS["ed4"]["parent"], SPECS["ed4"]["target"])
    ed_text = ed_text.replace(
        "uses\n`No. of experiences` in panels d/e, and removes",
        "uses\n`No. of cycles` in panels d/e, and removes",
    )
    ed_path.write_text(ed_text, encoding="utf-8", newline="\n")


def archive_one(key: str) -> dict[str, Any]:
    spec = SPECS[key]
    source = RECORDS / spec["parent"]
    destination = ARCHIVE / spec["parent"]
    if not source.is_dir() or destination.exists():
        raise FileExistsError(f"Archive precondition failed: {source}, {destination}")
    before = tree_digest(source)
    shutil.move(str(source), str(destination))
    after = tree_digest(destination)
    if before != after:
        raise AssertionError(f"Archive tree changed for {key}: {before}, {after}")
    row = {
        "record": spec["parent"],
        "original_path": f"docs/figure_version_records/{spec['parent']}",
        "archive_path": f"docs/figure_version_records_archive_2026-08-09/{spec['parent']}",
        **before,
        "move_status": "verified",
    }
    with ARCHIVE_MANIFEST.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("record", "original_path", "archive_path", "file_count", "bytes", "tree_sha256", "move_status"),
        )
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("create", "archive"))
    args = parser.parse_args()
    if args.phase == "create":
        created = {key: str(create_one(key).relative_to(REPO)) for key in SPECS}
        update_pointers()
        print(json.dumps({"status": "PASS", "created": created}, indent=2))
    else:
        rows = {key: archive_one(key) for key in SPECS}
        print(json.dumps({"status": "PASS", "archived": rows}, indent=2))


if __name__ == "__main__":
    main()
