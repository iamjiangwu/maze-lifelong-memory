"""Close the approved ED3 and ED10 append-only display successor records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs" / "figure_version_records"
ED3_PARENT = RECORDS / "extended_data_fig03_2026-08-11_round1_direct_entry_reconstruction"
ED3 = RECORDS / "extended_data_fig03_2026-08-12_round2_reward_path_terminology"
ED10_PARENT = RECORDS / "extended_data_fig10_2026-08-11_round10_upright_policy_labels"
ED10 = RECORDS / "extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order"
ED2 = RECORDS / "extended_data_fig02_2026-08-12_round23_preceding_turn"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def iter_relative_files(root: Path) -> list[tuple[str, Path]]:
    """Enumerate every file, including Windows paths beyond MAX_PATH."""
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


def relative_file_hashes(root: Path, folder: str) -> dict[str, str]:
    base = root / folder
    return {
        relative: sha256(path)
        for relative, path in iter_relative_files(base)
    }


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


def direct_numeric_cell_texts(svg_path: Path) -> list[ET.Element]:
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    axes = next(
        group
        for group in root.findall(".//svg:g", namespace)
        if group.attrib.get("id") == "axes_1"
    )
    cells: list[ET.Element] = []
    for text in axes.findall(".//svg:text", namespace):
        value = "".join(text.itertext()).strip()
        style = text.attrib.get("style", "")
        if re.fullmatch(r"-?\d+\.\d{2}", value) and re.search(
            r"font:\s+(?:700\s+)?8px", style
        ):
            cells.append(text)
    return cells


def policy_order(svg_path: Path) -> list[str]:
    return re.findall(
        r'id="policy_significance_([^"]+)"',
        svg_path.read_text(encoding="utf-8"),
    )


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


def make_comparison(
    parent_png: Path,
    successor_png: Path,
    destination: Path,
    right_title: str,
) -> None:
    with Image.open(parent_png) as before_source, Image.open(successor_png) as after_source:
        before = before_source.convert("RGB")
        after = after_source.convert("RGB")
    if before.size != after.size:
        raise AssertionError((before.size, after.size))
    target_width = 1500
    target_height = round(before.height * target_width / before.width)
    before = before.resize((target_width, target_height), Image.Resampling.LANCZOS)
    after = after.resize((target_width, target_height), Image.Resampling.LANCZOS)
    heading = 86
    gap = 28
    canvas = Image.new("RGB", (2 * target_width + gap, target_height + heading), "white")
    canvas.paste(before, (0, heading))
    canvas.paste(after, (target_width + gap, heading))
    font_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 40) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "Parent record", fill="black", font=font)
    draw.text((target_width + gap + 28, 20), right_title, fill="black", font=font)
    canvas.save(destination, dpi=(150, 150))


def write_record_hashes(root: Path) -> None:
    paths = [
        (relative, path)
        for relative, path in iter_relative_files(root)
        if relative != "RECORD_HASHES.csv"
    ]
    with (root / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        for relative, path in paths:
            writer.writerow(
                {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
            )


def validate_record_hashes(root: Path) -> None:
    with (root / "RECORD_HASHES.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        relative
        for relative, _ in iter_relative_files(root)
        if relative != "RECORD_HASHES.csv"
    }
    observed = {row["path"] for row in rows}
    if expected != observed:
        raise AssertionError({"missing": sorted(expected - observed), "extra": sorted(observed - expected)})
    for row in rows:
        normal_path = (root / row["path"]).resolve()
        path = Path(f"\\\\?\\{normal_path}") if os.name == "nt" else normal_path
        if sha256(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"Hash inventory mismatch: {row['path']}")


def close_ed3(created_utc: str) -> dict[str, object]:
    shutil.copyfile(ED3 / "ed03_version_record_600dpi.png", ED3 / "ed03_review_record_render.png")
    make_comparison(
        ED3_PARENT / "ed03_version_record_600dpi.png",
        ED3 / "ed03_version_record_600dpi.png",
        ED3 / "before_after_parent_vs_successor.png",
        "Successor: reader-facing reward-path terms",
    )
    review_path = ED3 / "REVIEW_REPORT.md"
    review_text = review_path.read_text(encoding="utf-8").replace("Latent day2", "Latent Day 2")
    review_path.write_text(review_text, encoding="utf-8")

    parent_svg = ED3_PARENT / "ed03_version_record.svg"
    svg = ED3 / "ed03_version_record.svg"
    svg_text = svg.read_text(encoding="utf-8")
    expected_text = {
        "Latent Day 2": 5,
        "Reward-path minus other dwell (s)": 1,
        "Mean reward-path junction dwell (s)": 1,
        "Reward-path approaches after reward": 1,
    }
    forbidden_text = [
        "Latent day2",
        "Direct-minus-other dwell (s)",
        "Mean direct-chain junction dwell (s)",
        "Post-port direct approach",
    ]
    geometry_equal = nontext_geometry(parent_svg) == nontext_geometry(svg)
    source_parity = relative_file_hashes(ED3_PARENT, "plot_ready") == relative_file_hashes(ED3, "plot_ready")
    vector = vector_qa(ED3 / "ed03_version_record.pdf", svg)
    checks = {
        "accepted_reader_facing_text_exact": all(svg_text.count(key) == value for key, value in expected_text.items()),
        "superseded_reader_facing_text_absent": not any(term in svg_text for term in forbidden_text),
        "nontext_svg_geometry_exact_parent": geometry_equal,
        "plot_ready_sources_byte_exact_parent": source_parity,
        "pdf_single_page_vector_only": vector["pdf_pages"] == 1 and vector["pdf_embedded_images"] == 0,
        "svg_live_text_without_images_or_external_links": vector["svg_editable_text"] and vector["svg_embedded_images"] == 0 and not vector["svg_external_links"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = f"""# {ED3.name}

This append-only successor supersedes `{ED3_PARENT.name}` after author approval.
Only reader-facing terminology changed: `Latent Day 2` is standardized, and
compressed route wording is replaced by `reward-path` wording. All plotted data,
statistics, axes, non-text SVG geometry and plot-ready sources are unchanged.

The AF23 direct-entry reconstruction and all inherited scientific provenance remain
in force. `DISPLAY_REVISION_MANIFEST.json`, `ED03_ASSEMBLY_MANIFEST.json`, the frozen
renderers under `source_code/`, and `RECORD_HASHES.csv` close the revision.
"""
    (ED3 / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    revision = {
        "schema": "ed03-reward-path-terminology-successor/v1",
        "status": "PASS",
        "record": ED3.name,
        "supersedes": ED3_PARENT.name,
        "created_utc": created_utc,
        "changes": [
            "Latent day2 -> Latent Day 2",
            "Direct-minus-other dwell -> Reward-path minus other dwell",
            "Mean direct-chain junction dwell -> Mean reward-path junction dwell",
            "Post-port direct approach -> Reward-path approaches after reward",
        ],
        "expected_text_counts": {key: svg_text.count(key) for key in expected_text},
        "scientific_parity": checks,
        "vector_qa": vector,
        "parent_exports": [
            file_record(ED3_PARENT / name, ED3_PARENT)
            for name in ("ed03_version_record.svg", "ed03_version_record.pdf")
        ],
        "successor_exports": [
            file_record(ED3 / name, ED3)
            for name in (
                "ed03_version_record.svg",
                "ed03_version_record.pdf",
                "ed03_version_record_600dpi.png",
                "ed03_version_record_600dpi.tiff",
            )
        ],
    }
    (ED3 / "DISPLAY_REVISION_MANIFEST.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    promotion = {
        "schema": "ed03-display-successor-promotion-audit/v1",
        "status": "PASS",
        "record": ED3.name,
        "supersedes": ED3_PARENT.name,
        "parent_record_preserved": True,
        "display_revision_manifest": "DISPLAY_REVISION_MANIFEST.json",
    }
    (ED3 / "AUDIT_PROMOTION.json").write_text(
        json.dumps(promotion, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "schema": "ed03-display-successor-record-validation/v1",
        "status": "PASS",
        "record": ED3.name,
        "validated_utc": created_utc,
        "checks": checks,
    }
    (ED3 / "RECORD_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (ED3 / "PROMOTION_COMPLETE.txt").write_text(
        f"PASS\nrecord={ED3.name}\ncreated_utc={created_utc}\n", encoding="utf-8"
    )
    shutil.copyfile(__file__, ED3 / "source_code" / Path(__file__).name)
    write_record_hashes(ED3)
    validate_record_hashes(ED3)
    return {"record": ED3.as_posix(), "checks": checks}


def close_ed10(created_utc: str) -> dict[str, object]:
    with Image.open(ED10 / "ed10_version_record_600dpi.png") as source:
        source.convert("RGB").resize((2250, 2250), Image.Resampling.LANCZOS).save(
            ED10 / "ed10_version_record_300dpi.png", dpi=(300, 300)
        )
    shutil.copyfile(ED10 / "ED10_CAPTION_NOTES.md", ED10 / "CAPTION_NOTES.md")
    make_comparison(
        ED10_PARENT / "ed10_version_record_300dpi.png",
        ED10 / "ed10_version_record_300dpi.png",
        ED10 / "before_after_parent_vs_successor.png",
        "Successor: uniform black values + ED2 order",
    )

    svg = ED10 / "ed10_version_record.svg"
    cells = direct_numeric_cell_texts(svg)
    values = ["".join(cell.itertext()).strip() for cell in cells]
    cell_styles = [cell.attrib.get("style", "") for cell in cells]
    order = policy_order(svg)
    ed2_renderer = (
        ED2 / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
    ).read_text(encoding="utf-8")
    source_parity = all(
        relative_file_hashes(ED10_PARENT, folder) == relative_file_hashes(ED10, folder)
        for folder in ("plot_ready", "source_authority", "main_figure_authority")
    )
    panel_b_parity = sha256(
        ED10_PARENT / "individual_panels" / "b_variation_partition.png"
    ) == sha256(ED10 / "individual_panels" / "b_variation_partition.png")
    renderer_validation = json.loads((ED10 / "AUDIT_VALIDATION.json").read_text(encoding="utf-8"))
    vector = vector_qa(ED10 / "ed10_version_record.pdf", svg)
    checks = {
        "panel_a_exactly_18_numeric_cells": len(cells) == 18,
        # Matplotlib omits an explicit fill declaration for default black.
        "panel_a_all_values_black": len(cells) == 18
        and all("fill:" not in style or "fill: #000000" in style for style in cell_styles),
        "panel_a_all_values_normal_weight": len(cells) == 18 and all("font: 700" not in style and "font-weight" not in style for style in cell_styles),
        "panel_a_values_exact_parent": sorted(values) == sorted(
            "".join(cell.itertext()).strip()
            for cell in direct_numeric_cell_texts(ED10_PARENT / "ed10_version_record.svg")
        ),
        "panel_c_matches_ed2_policy_order": order == ["Bf", "Ba", "Lf", "Lo_depth_weighted"] and 'POLICY_ORDER = ("Bf", "Ba", "Lf", "Lo")' in ed2_renderer,
        "scientific_authorities_byte_exact_parent": source_parity,
        "unrequested_panel_b_pixel_exact_parent": panel_b_parity,
        "maintained_renderer_validation_pass": renderer_validation.get("status") == "PASS",
        "pdf_single_page_vector_only": vector["pdf_pages"] == 1 and vector["pdf_embedded_images"] == 0,
        "svg_live_text_without_images_or_external_links": vector["svg_editable_text"] and vector["svg_embedded_images"] == 0 and not vector["svg_external_links"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest_text = f"""# {ED10.name}

This append-only successor supersedes `{ED10_PARENT.name}` after author approval.
Panel a retains all 18 descriptive Pearson correlations but renders every value in
black, normal-weight text. Panel c follows the Extended Data Figure 2 policy order:
`P_SF`, `P_SA`, `P_BF`, depth-weighted `P_BS`. Panel b, quantitative values,
statistics, cohorts, exclusions and scientific authorities are unchanged.

See `DISPLAY_REVISION_MANIFEST.json`, `AUDIT_VALIDATION.json`, the frozen renderer
under `source_code/`, and `RECORD_HASHES.csv` for closure.
"""
    (ED10 / "MANIFEST.md").write_text(manifest_text, encoding="utf-8")
    revision = {
        "schema": "ed10-black-correlation-values-ed2-policy-order-successor/v1",
        "status": "PASS",
        "record": ED10.name,
        "record_revision": 11,
        "supersedes": ED10_PARENT.name,
        "created_utc": created_utc,
        "changes": {
            "panel_a": "all 18 correlation values use black normal-weight text",
            "panel_c": ["P_SF", "P_SA", "P_BF", "depth-weighted P_BS"],
        },
        "displayed_panel_a_values": values,
        "scientific_parity": checks,
        "vector_qa": vector,
        "parent_exports": [
            file_record(ED10_PARENT / name, ED10_PARENT)
            for name in ("ed10_version_record.svg", "ed10_version_record.pdf")
        ],
        "successor_exports": [
            file_record(ED10 / name, ED10)
            for name in (
                "ed10_version_record.svg",
                "ed10_version_record.pdf",
                "ed10_version_record_300dpi.png",
                "ed10_version_record_600dpi.png",
                "ed10_version_record_600dpi.tiff",
            )
        ],
    }
    (ED10 / "DISPLAY_REVISION_MANIFEST.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    top_manifest = {
        "schema": "ed10-round11-black-correlations-ed2-policy-order-version-record/v1",
        "status": "ACCEPTED",
        "record": ED10.name,
        "record_revision": 11,
        "supersedes": ED10_PARENT.name,
        "accepted_utc": created_utc,
        "figure": "Extended Data Figure 10",
        "panel_letters": ["a", "b", "c"],
        "core_conclusion": "Across the fixed-neuron cohort, neural measures covary broadly with later behavior; whole maze decoding constrains reward-specific interpretation, while the separate clean36 panel reports the fixed level-policy bridge.",
        "current_display_refinement": revision["changes"],
        "scientific_scope": {
            "panels_a_b": "30 eligible session pairs / 14 mice / K=166; technical draws averaged within pair",
            "panel_c": "36 eligible session pairs / 14 mice; separate clean36 family",
            "manual_neuron_mask": "none",
        },
        "display_contract": {
            "colorbar_title": "Pearson correlation",
            "panel_a_label_color": "black",
            "panel_a_label_weight": "normal",
            "panel_a_cell_color": "black",
            "panel_a_cell_weight": "normal",
            "panel_a_cell_count": 18,
            "panel_c_policy_order": ["Bf", "Ba", "Lf", "Lo_depth_weighted"],
            "panel_c_annotations": ["n.s.", "n.s.", "n.s.", "***"],
        },
        "root_export_sha256": {
            name: sha256(ED10 / filename)
            for name, filename in {
                "svg": "ed10_version_record.svg",
                "pdf": "ed10_version_record.pdf",
                "png_300dpi": "ed10_version_record_300dpi.png",
                "png_600dpi": "ed10_version_record_600dpi.png",
                "tiff_600dpi": "ed10_version_record_600dpi.tiff",
            }.items()
        },
        "record_validation": "RECORD_VALIDATION.json",
        "display_revision_manifest": "DISPLAY_REVISION_MANIFEST.json",
    }
    (ED10 / "ED10_VERSION_RECORD_MANIFEST.json").write_text(
        json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8"
    )
    cross_figure = {
        "schema": "cross-figure-refinement-record/v2",
        "status": "PASS",
        "figure": "extended_data_10",
        "record": ED10.name,
        "supersedes": ED10_PARENT.name,
        "promoted_utc": created_utc,
        "authorized_changes": [
            "panel-a correlation values standardized to black normal-weight text",
            "panel-c policy ordering aligned to Extended Data Figure 2",
        ],
        "preserved": [
            "all 18 displayed correlation values",
            "panel b commonality",
            "cohort 30 pairs / 14 mice / K=166",
            "panel-c data, statistics and annotations",
        ],
    }
    (ED10 / "CROSS_FIGURE_REFINEMENT_MANIFEST.json").write_text(
        json.dumps(cross_figure, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "schema": "ed10-display-successor-record-validation/v1",
        "status": "PASS",
        "record": ED10.name,
        "validated_utc": created_utc,
        "focused_validation": "AUDIT_VALIDATION.json",
        "checks": checks,
    }
    (ED10 / "RECORD_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (ED10 / "REBUILD_README.md").write_text(
        f"# Rebuild {ED10.name}\n\nRun the frozen `source_code/render_ed10_slide12_powerpoint_matched.py` against the record-local authorities, then apply the root export names documented in `DISPLAY_REVISION_MANIFEST.json`.\n",
        encoding="utf-8",
    )
    (ED10 / "PROMOTION_COMPLETE.txt").write_text(
        f"PASS\nrecord={ED10.name}\ncreated_utc={created_utc}\n", encoding="utf-8"
    )
    shutil.copyfile(__file__, ED10 / "source_code" / Path(__file__).name)
    shutil.copyfile(
        REPO / "scripts" / "validate_current_figure_records.py",
        ED10 / "source_code" / "validate_current_figure_records.py",
    )
    shutil.copyfile(
        REPO / "scripts" / "validate_all_figure_refinement_round2_records.py",
        ED10 / "source_code" / "validate_all_figure_refinement_round2_records.py",
    )
    write_record_hashes(ED10)
    validate_record_hashes(ED10)
    return {"record": ED10.as_posix(), "checks": checks}


def main() -> None:
    created_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "PASS",
        "created_utc": created_utc,
        "ed3": close_ed3(created_utc),
        "ed10": close_ed10(created_utc),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
