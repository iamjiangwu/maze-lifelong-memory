"""Build and validate the ED3 lowercase latent-day label successor record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs" / "figure_version_records"
PARENT = RECORDS / "extended_data_fig03_2026-08-12_round2_reward_path_terminology"
SUCCESSOR = RECORDS / "extended_data_fig03_2026-08-13_round3_latent_day_labels"
RENDERER = REPO / "scripts" / "render_ed03_reward_path_fluency_assembly.py"
FLUENCY_RENDERER = REPO / "scripts" / "render_ed04_fluency_story_current.py"

ROOT_EXPORTS = {
    "svg": "ed03_version_record.svg",
    "pdf": "ed03_version_record.pdf",
    "png": "ed03_version_record_600dpi.png",
    "tiff": "ed03_version_record_600dpi.tiff",
}
ASSEMBLY_EXPORTS = {
    "svg": "ed03_reward_path_fluency_assembly.svg",
    "pdf": "ed03_reward_path_fluency_assembly.pdf",
    "png": "ed03_reward_path_fluency_assembly.png",
    "tiff": "ed03_reward_path_fluency_assembly.tiff",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_relative_files(root: Path) -> list[tuple[str, Path]]:
    resolved = root.resolve()
    walk_root = Path(f"\\\\?\\{resolved}") if os.name == "nt" else resolved
    rows: list[tuple[str, Path]] = []
    for directory, _, names in os.walk(walk_root):
        directory_path = Path(directory)
        for name in names:
            extended_path = directory_path / name
            normal_text = str(extended_path)
            if normal_text.startswith("\\\\?\\"):
                normal_text = normal_text[4:]
            relative = Path(normal_text).relative_to(resolved).as_posix()
            rows.append((relative, extended_path))
    return sorted(rows, key=lambda item: item[0])


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def relative_file_hashes(root: Path, folder: str) -> dict[str, str]:
    base = root / folder
    return {relative: sha256(path) for relative, path in iter_relative_files(base)}


def nontext_geometry(path: Path) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    root = ET.parse(path).getroot()
    rows: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"text", "tspan", "title", "desc", "metadata"}:
            continue
        if tag not in {
            "path",
            "use",
            "line",
            "polyline",
            "polygon",
            "rect",
            "circle",
            "ellipse",
        }:
            continue
        attrs = tuple(
            sorted(
                (key.rsplit("}", 1)[-1], value)
                for key, value in node.attrib.items()
                if key.rsplit("}", 1)[-1] not in {"id", "href", "clip-path"}
            )
        )
        rows.append((tag, attrs))
    return rows


def vector_qa(pdf_path: Path, svg_path: Path) -> dict[str, object]:
    document = fitz.open(pdf_path)
    pages = len(document)
    images = sum(len(page.get_images(full=True)) for page in document)
    document.close()
    svg = svg_path.read_text(encoding="utf-8")
    return {
        "pdf_pages": pages,
        "pdf_embedded_images": images,
        "svg_embedded_images": svg.count("<image"),
        "svg_editable_text": "<text" in svg,
        "svg_external_links": "file://" in svg
        or bool(re.search(r'(?:href|xlink:href)=["\']https?://', svg)),
    }


def make_comparison(parent_png: Path, successor_png: Path, destination: Path) -> None:
    with Image.open(parent_png) as parent_source, Image.open(successor_png) as successor_source:
        parent = parent_source.convert("RGB")
        successor = successor_source.convert("RGB")
    if parent.size != successor.size:
        raise AssertionError((parent.size, successor.size))
    target_width = 1500
    target_height = round(parent.height * target_width / parent.width)
    parent = parent.resize((target_width, target_height), Image.Resampling.LANCZOS)
    successor = successor.resize((target_width, target_height), Image.Resampling.LANCZOS)
    heading = 86
    gap = 28
    canvas = Image.new("RGB", (2 * target_width + gap, target_height + heading), "white")
    canvas.paste(parent, (0, heading))
    canvas.paste(successor, (target_width + gap, heading))
    font_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 40) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "Parent: Latent Day 2", fill="black", font=font)
    draw.text((target_width + gap + 28, 20), "Successor: Latent day2", fill="black", font=font)
    canvas.save(destination, dpi=(150, 150))


def write_record_hashes(root: Path) -> None:
    rows = [
        (relative, path)
        for relative, path in iter_relative_files(root)
        if relative != "RECORD_HASHES.csv"
    ]
    with (root / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        for relative, path in rows:
            writer.writerow(
                {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
            )


def validate_record_hashes(root: Path) -> None:
    with (root / "RECORD_HASHES.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        relative for relative, _ in iter_relative_files(root) if relative != "RECORD_HASHES.csv"
    }
    observed = {row["path"] for row in rows}
    if expected != observed:
        raise AssertionError(
            {"missing": sorted(expected - observed), "extra": sorted(observed - expected)}
        )
    for row in rows:
        normal_path = (root / row["path"]).resolve()
        path = Path(f"\\\\?\\{normal_path}") if os.name == "nt" else normal_path
        if sha256(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"Hash inventory mismatch: {row['path']}")


def checks_for(root: Path) -> tuple[dict[str, bool], dict[str, object]]:
    svg = root / ROOT_EXPORTS["svg"]
    svg_text = svg.read_text(encoding="utf-8")
    vector = vector_qa(root / ROOT_EXPORTS["pdf"], svg)
    checks = {
        "latent_day2_exactly_five_visible_labels": svg_text.count("Latent day2") == 5,
        "superseded_latent_Day_2_absent": "Latent Day 2" not in svg_text,
        "reward_path_minus_other_preserved": svg_text.count("Reward-path minus other dwell (s)") == 1,
        "reward_path_junction_preserved": svg_text.count("Mean reward-path junction dwell (s)") == 1,
        "reward_path_approach_preserved": svg_text.count("Reward-path approaches after reward") == 1,
        "nontext_svg_geometry_exact_parent": nontext_geometry(PARENT / ROOT_EXPORTS["svg"])
        == nontext_geometry(svg),
        "plot_ready_sources_byte_exact_parent": relative_file_hashes(PARENT, "plot_ready")
        == relative_file_hashes(root, "plot_ready"),
        "source_validation_byte_exact_parent": relative_file_hashes(PARENT, "source_validation")
        == relative_file_hashes(root, "source_validation"),
        "assembly_and_root_exports_identical": all(
            sha256(root / ROOT_EXPORTS[key]) == sha256(root / ASSEMBLY_EXPORTS[key])
            for key in ROOT_EXPORTS
        ),
        "renderer_validation_pass": json.loads(
            (root / "ED03_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
        )["validation_status"]
        == "pass",
        "pdf_single_page_vector_only": vector["pdf_pages"] == 1
        and vector["pdf_embedded_images"] == 0,
        "svg_live_text_without_images_or_external_links": vector["svg_editable_text"]
        and vector["svg_embedded_images"] == 0
        and not vector["svg_external_links"],
    }
    return checks, vector


def build() -> None:
    if SUCCESSOR.exists():
        raise FileExistsError(f"Refusing to overwrite append-only successor: {SUCCESSOR}")
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    shutil.copytree(PARENT, SUCCESSOR)
    subprocess.run(
        [sys.executable, str(RENDERER), "--output-dir", str(SUCCESSOR)],
        cwd=REPO,
        check=True,
    )
    for key, source_name in ASSEMBLY_EXPORTS.items():
        shutil.copyfile(SUCCESSOR / source_name, SUCCESSOR / ROOT_EXPORTS[key])
    shutil.copyfile(
        SUCCESSOR / ROOT_EXPORTS["png"], SUCCESSOR / "ed03_review_record_render.png"
    )
    make_comparison(
        PARENT / ROOT_EXPORTS["png"],
        SUCCESSOR / ROOT_EXPORTS["png"],
        SUCCESSOR / "before_after_parent_vs_successor.png",
    )

    review_path = SUCCESSOR / "REVIEW_REPORT.md"
    review_path.write_text(
        review_path.read_text(encoding="utf-8").replace("Latent Day 2", "Latent day2"),
        encoding="utf-8",
    )
    source_code = SUCCESSOR / "source_code"
    shutil.copyfile(RENDERER, source_code / RENDERER.name)
    shutil.copyfile(FLUENCY_RENDERER, source_code / FLUENCY_RENDERER.name)
    shutil.copyfile(__file__, source_code / Path(__file__).name)

    created_utc = datetime.now(timezone.utc).isoformat()
    checks, vector = checks_for(SUCCESSOR)
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = f"""# {SUCCESSOR.name}

This append-only successor supersedes `{PARENT.name}` after author approval.
The only reader-facing change is the exact label substitution `Latent Day 2` to
`Latent day2` in panels a--d and f. Reward-path terminology, plotted data,
statistics, axes, non-text SVG geometry and plot-ready sources are unchanged.

`DISPLAY_REVISION_MANIFEST.json`, `ED03_ASSEMBLY_MANIFEST.json`, the frozen
renderers under `source_code/`, and `RECORD_HASHES.csv` close the revision.
"""
    (SUCCESSOR / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    revision = {
        "schema": "ed03-latent-day-label-successor/v1",
        "status": "PASS",
        "record": SUCCESSOR.name,
        "record_revision": 3,
        "supersedes": PARENT.name,
        "created_utc": created_utc,
        "authorized_change": "Latent Day 2 -> Latent day2 in five visible labels",
        "preserved": [
            "reward-path terminology",
            "all plotted values and statistics",
            "all axes and non-text SVG geometry",
            "plot-ready sources and source validation",
        ],
        "scientific_parity": checks,
        "vector_qa": vector,
        "parent_exports": [
            file_record(PARENT / name, PARENT)
            for name in (ROOT_EXPORTS["svg"], ROOT_EXPORTS["pdf"])
        ],
        "successor_exports": [
            file_record(SUCCESSOR / name, SUCCESSOR) for name in ROOT_EXPORTS.values()
        ],
    }
    (SUCCESSOR / "DISPLAY_REVISION_MANIFEST.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "schema": "ed03-latent-day-label-record-validation/v1",
        "status": "PASS",
        "record": SUCCESSOR.name,
        "validated_utc": created_utc,
        "checks": checks,
    }
    (SUCCESSOR / "RECORD_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    promotion = {
        "schema": "ed03-latent-day-label-promotion-audit/v1",
        "status": "PASS",
        "record": SUCCESSOR.name,
        "supersedes": PARENT.name,
        "parent_record_preserved": True,
        "display_revision_manifest": "DISPLAY_REVISION_MANIFEST.json",
    }
    (SUCCESSOR / "AUDIT_PROMOTION.json").write_text(
        json.dumps(promotion, indent=2) + "\n", encoding="utf-8"
    )
    (SUCCESSOR / "PROMOTION_COMPLETE.txt").write_text(
        f"PASS\nrecord={SUCCESSOR.name}\ncreated_utc={created_utc}\n", encoding="utf-8"
    )
    write_record_hashes(SUCCESSOR)
    validate_record_hashes(SUCCESSOR)


def validate_only() -> None:
    if not SUCCESSOR.is_dir():
        raise FileNotFoundError(SUCCESSOR)
    checks, vector = checks_for(SUCCESSOR)
    validate_record_hashes(SUCCESSOR)
    manifest = json.loads((SUCCESSOR / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("checks") != checks:
        raise AssertionError("RECORD_VALIDATION.json does not match live checks")
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps({"status": "PASS", "record": SUCCESSOR.name, "checks": checks, "vector_qa": vector}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return
    build()
    validate_only()


if __name__ == "__main__":
    main()
