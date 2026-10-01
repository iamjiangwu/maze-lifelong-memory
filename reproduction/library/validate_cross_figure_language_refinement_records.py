"""Validate the 2026-08-11 cross-figure language and typography refinement.

This validator is intentionally read-only with respect to immutable figure
records.  It checks the promoted display semantics, font identity, measured
label placement, collision clearance, scientific-source parity, pointer
selection, and caption terminology.  The aggregate verdict is written beside
the current-record ledgers, outside the immutable records themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import fitz
from PIL import Image, ImageChops


REPO = Path(__file__).resolve().parents[1]
RECORD_ROOT = REPO / "docs/figure_version_records"
ARCHIVE_ROOT = REPO / "docs/figure_version_records_archive_2026-08-09"
CAPTION_ROOT = REPO / "docs/manuscript_figure_captions_2026-08-10"
DEFAULT_OUTPUT = RECORD_ROOT / "CROSS_FIGURE_LANGUAGE_REFINEMENT_VALIDATION.json"

RECORDS = {
    "main3": (
        "main_figure3_2026-08-11_round3_upright_day_i",
        "main_figure3_2026-08-11_round2_timing_schematic_open_terms",
    ),
    "main4": (
        "main_figure4_2026-08-11_round3_upright_i_density_ticks",
        "main_figure4_2026-08-11_round2_place_maps_label",
    ),
    "ed1": (
        "extended_data_fig01_2026-08-11_round10_open_terms",
        "extended_data_fig01_2026-08-09_round9_h_legend",
    ),
    "ed2": (
        "extended_data_fig02_2026-08-11_round19_upright_policy_complete",
        "extended_data_fig02_2026-08-11_round18_upright_policy_labels",
    ),
    "ed6": (
        "extended_data_fig06_2026-08-11_round3_maze_tile_label",
        "extended_data_fig06_2026-08-11_round2_open_terms",
    ),
    "ed7": (
        "extended_data_fig07_2026-08-11_round12_open_long_gap",
        "extended_data_fig07_2026-08-09_round11_concise_row_titles",
    ),
    "ed8": (
        "extended_data_fig08_2026-08-11_round10_horizontal_ticks",
        "extended_data_fig08_2026-08-11_round9_open_terms",
    ),
}

FORBIDDEN_VISIBLE = ("maze-tile", "long-bout", "long-gap", "reward-directed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_record(name: str) -> Path:
    for root in (RECORD_ROOT, ARCHIVE_ROOT):
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(name)


def find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern} under {root}, found {len(paths)}")
    return paths[0]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def same_tree(target: Path, parent: Path, relative: str) -> bool:
    return tree_hashes(target / relative) == tree_hashes(parent / relative)


def same_pixels(left: Path, right: Path) -> bool:
    with Image.open(left) as first, Image.open(right) as second:
        if first.size != second.size:
            return False
        difference = ImageChops.difference(first.convert("RGBA"), second.convert("RGBA"))
        return difference.getbbox() is None


def pdf_spans(path: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    direction = tuple(line.get("dir", (1.0, 0.0)))
                    for span in line.get("spans", []):
                        spans.append(
                            {
                                **span,
                                "page": page_index,
                                "direction": direction,
                            }
                        )
    return spans


def pdf_text(path: Path) -> str:
    with fitz.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def pdf_image_count(path: Path) -> int:
    with fitz.open(path) as document:
        return sum(len(page.get_images(full=True)) for page in document)


def embedded_image_hashes(svg: Path) -> list[str]:
    text = svg.read_text(encoding="utf-8")
    payloads = re.findall(
        r'''(?:href|xlink:href)=["']data:image/[^;]+;base64,([^"']+)["']''',
        text,
        flags=re.IGNORECASE,
    )
    return [hashlib.sha256(payload.encode("ascii")).hexdigest() for payload in payloads]


def svg_shape_signature(path: Path) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """Return non-text geometry while ignoring volatile IDs and clip-path names."""

    root = ET.parse(path).getroot()
    signature: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in {"path", "rect", "circle", "line", "polyline", "polygon", "image", "use"}:
            continue
        attributes = {
            key.rsplit("}", 1)[-1]: value
            for key, value in element.attrib.items()
            if key.rsplit("}", 1)[-1] not in {"id", "clip-path"}
        }
        href = attributes.get("href", "")
        if href.startswith("#"):
            attributes["href"] = "#REF"
        elif href.startswith("data:"):
            attributes["href"] = hashlib.sha256(href.encode("ascii")).hexdigest()
        signature.append((tag, tuple(sorted(attributes.items()))))
    return signature


def close(value: float, expected: float, tolerance: float = 1e-4) -> bool:
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)


def record_export_checks(record: Path, expected_images: int) -> dict[str, bool]:
    svg = find_one(record, "*_version_record.svg")
    pdf = find_one(record, "*_version_record.pdf")
    png = find_one(record, "*_version_record_600dpi.png")
    tiff = find_one(record, "*_version_record_600dpi.tiff")
    validation = json.loads((record / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    with fitz.open(pdf) as document:
        one_page = document.page_count == 1
    with Image.open(png) as image:
        png_ok = image.width > 1000 and image.height > 1000
    with Image.open(tiff) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        tiff_ok = all(abs(float(value) - 600.0) < 1.0 for value in dpi)
    return {
        "root_svg_selectable_text": svg.read_text(encoding="utf-8").count("<text") > 0,
        "root_pdf_one_page": one_page,
        "root_png_nontrivial": png_ok,
        "root_tiff_600dpi": tiff_ok,
        "root_pdf_image_policy": pdf_image_count(pdf) == expected_images,
        "record_validation_pass": str(validation.get("status", "")).lower() == "pass",
        "record_validation_identity": validation.get("record") == record.name,
    }


def main3_checks(target: Path, parent: Path) -> dict[str, bool]:
    pdf = target / "figure3_version_record.pdf"
    spans = pdf_spans(pdf)
    labels = [span for span in spans if span["text"] in {"Day i", "Day i + 1"}]
    literal_ok = len(labels) >= 2 and all(
        "arial" in span["font"].lower()
        and "italic" not in span["font"].lower()
        and "oblique" not in span["font"].lower()
        for span in labels
    )
    unchanged_panels = []
    for source in sorted((target / "individual_panels").glob("*.svg")):
        if "panel_b_" in source.name:
            continue
        peer = parent / "individual_panels" / source.name
        if peer.is_file():
            unchanged_panels.append(
                svg_shape_signature(source) == svg_shape_signature(peer)
            )
    return {
        "main3_source_data_exact": same_tree(target, parent, "source_data"),
        "main3_day_i_literal_upright_arial": literal_ok,
        "main3_other_panel_pixels_exact": bool(unchanged_panels) and all(unchanged_panels),
    }


def main4_checks(target: Path, parent: Path) -> dict[str, bool]:
    spans = pdf_spans(target / "individual_panels/figure4_panel_a_d1.pdf")
    labels = [span for span in spans if "Day i" in span["text"]]
    literal_ok = len(labels) >= 2 and all(
        "arial" in span["font"].lower()
        and "italic" not in span["font"].lower()
        and "oblique" not in span["font"].lower()
        for span in labels
    )

    e_pdf = target / "individual_panels/figure4_panel_e_d1.pdf"
    with fitz.open(e_pdf) as document:
        words = document[0].get_text("words")
    density_zeros = [
        word
        for word in words
        if word[4] == "0"
        and (115.0 <= float(word[0]) <= 116.5 or 277.0 <= float(word[0]) <= 278.5)
    ]
    clearances: list[float] = []
    for density_zero in density_zeros:
        candidates = [
            word
            for word in words
            if word[2] < density_zero[0]
            and abs(word[1] - density_zero[1]) < 0.2
            and word[4] in {"0", "21"}
        ]
        if candidates:
            heatmap_tick = max(candidates, key=lambda word: word[2])
            clearances.append(float(density_zero[0] - heatmap_tick[2]))

    unchanged = []
    for role in "bcdf":
        for suffix in ("png", "tiff"):
            name = f"figure4_panel_{role}_d1.{suffix}"
            unchanged.append(
                sha256(target / "individual_panels" / name)
                == sha256(parent / "individual_panels" / name)
            )
    target_payloads = embedded_image_hashes(
        target / "individual_panels/figure4_panel_e_d1.svg"
    )
    parent_payloads = embedded_image_hashes(
        parent / "individual_panels/figure4_panel_e_d1.svg"
    )
    return {
        "main4_plot_ready_exact": same_tree(target, parent, "plot_ready"),
        "main4_day_i_literal_upright_arial": literal_ok,
        "main4_bcdf_rasters_exact": all(unchanged),
        "main4_panel_e_heatmap_payloads_exact": len(target_payloads) == 5
        and target_payloads == parent_payloads,
        "main4_panel_e_density_origin_integer": len(clearances) == 2
        and min(clearances) >= 0.75,
    }


def ed1_checks(target: Path, parent: Path) -> dict[str, bool]:
    svg = (target / "ed01_version_record.svg").read_text(encoding="utf-8")
    return {
        "ed1_drawn_values_exact": same_tree(target, parent, "drawn_values"),
        "ed1_plot_ready_exact": same_tree(target, parent, "plot_ready"),
        "ed1_maze_tile_open_five": svg.count("Maze tile transitions") == 5,
        "ed1_time_in_maze_label": "Time in maze (s)" in svg and "Maze time (s)" not in svg,
        "ed1_long_bout_open": "First long bout time (s)" in svg,
        "ed1_reward_directed_open": "Reward directed" in svg,
    }


def ed2_checks(target: Path, parent: Path) -> dict[str, bool]:
    expected_sizes = {
        "a": [6.2] * 4,
        "b": [6.0] * 4,
        "c": [6.0] * 4,
        "d": [5.4] * 4,
        "e": [5.4] * 4 + [6.0],
        "g": [6.0, 6.0],
        "h": [5.4],
        "i": [6.2],
    }
    sizes_ok = True
    fonts_ok = True
    for role, expected in expected_sizes.items():
        panel = find_one(target / "panels", f"ed02{role}_*.pdf")
        relevant = [
            span
            for span in pdf_spans(panel)
            if "P" in span["text"] or "Δ" in span["text"]
        ]
        observed = sorted(round(float(span["size"]), 1) for span in relevant)
        sizes_ok &= observed == sorted(expected)
        fonts_ok &= all(
            "arial" in span["font"].lower()
            and "italic" not in span["font"].lower()
            and "oblique" not in span["font"].lower()
            for span in relevant
        )

    a_spans = pdf_spans(find_one(target / "panels", "ed02a_*.pdf"))
    preceding = [span for span in a_spans if span["text"] == "preceding"]
    turn = [span for span in a_spans if span["text"] == "turn"]
    annotation_ok = (
        len(preceding) == 1
        and len(turn) == 1
        and close(float(preceding[0]["bbox"][0]), 49.1382, 0.02)
        and close(float(turn[0]["bbox"][1]), 10.6404, 0.02)
        and float(turn[0]["bbox"][1]) > float(preceding[0]["bbox"][1])
    )

    master_svg = (target / "ed02_version_record.svg").read_text(encoding="utf-8")
    y_first = re.search(
        r'transform="matrix\([^\"]+\s([0-9.]+)\)"[^>]*>.*Return day &#x2212; Naive day',
        master_svg,
    )
    y_second = re.search(
        r'transform="matrix\([^\"]+\s([0-9.]+)\)"[^>]*>.*Second day &#x2212; Return day',
        master_svg,
    )
    label_positions_ok = bool(y_first and y_second) and close(
        float(y_first.group(1)), 452.66868
    ) and close(float(y_second.group(1)), 460.43855)

    manifest = json.loads((target / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    annotation_contract = manifest["panel_a_display_contract_round12"]["preceding_turn"]
    return {
        "ed2_analysis_inputs_exact": same_tree(target, parent, "analysis_inputs"),
        "ed2_analysis_provenance_exact": same_tree(target, parent, "analysis_provenance"),
        "ed2_plot_ready_exact": same_tree(target, parent, "plot_ready"),
        "ed2_all_policy_p_upright_arial": fonts_ok,
        "ed2_role_font_sizes_exact": sizes_ok,
        "ed2_preceding_turn_two_lines_positioned": annotation_ok,
        "ed2_preceding_turn_contract_current": annotation_contract.get("text")
        == "preceding\nturn"
        and annotation_contract.get("xy") == [2.9, 3.82]
        and bool(annotation_contract.get("must_not_overlap_grid_or_arrow")),
        "ed2_panel_g_lower_labels_exact_raise": label_positions_ok,
    }


def ed6_checks(target: Path, parent: Path) -> dict[str, bool]:
    svg = (target / "ed06_version_record.svg").read_text(encoding="utf-8")
    return {
        "ed6_gate_authority_exact": same_tree(target, parent, "gate_authority"),
        "ed6_gate_reference_exact": same_tree(target, parent, "gate_reference"),
        "ed6_plot_ready_exact": same_tree(target, parent, "plot_ready"),
        "ed6_maze_tile_open": "Maze tile transitions" in svg,
    }


def ed7_checks(target: Path, parent: Path) -> dict[str, bool]:
    svg = (target / "ed07_version_record.svg").read_text(encoding="utf-8")
    manifest = json.loads(
        (target / "ED07_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    display_manifest = json.loads(
        (target / "ED07_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
    )
    display = display_manifest["display_contract"]
    top_exact: list[bool] = []
    for suffix in ("png", "tiff"):
        target_top = find_one(target / "component_exports", f"*abc*12.{suffix}")
        parent_top = find_one(parent / "component_exports", f"*abc*11.{suffix}")
        top_exact.append(same_pixels(target_top, parent_top))
    return {
        "ed7_plot_ready_exact": same_tree(target, parent, "plot_ready"),
        "ed7_source_authority_exact": same_tree(target, parent, "source_authority"),
        "ed7_palette_authority_exact": same_tree(target, parent, "palette_authority"),
        "ed7_long_gap_open": "Long gap correlation" in svg,
        "ed7_top_component_pixels_exact": all(top_exact),
        "ed7_manifest_round12_current": manifest.get("accepted_display_round") == 12
        and manifest.get("bottom_y_axis_label") == "Long gap correlation"
        and manifest.get("display_contract", {}).get("bottom_y_axis_label")
        == "Long gap correlation",
        "ed7_display_contract_current": display.get("bottom_y_axis_label")
        == "Long gap correlation",
        "ed7_exclusions_and_mask_locked": len(
            display_manifest.get("excluded_from_all_ed7_primary_analyses", [])
        )
        == 6
        and display_manifest.get("manual_mask_profile")
        == "none for every displayed source row",
    }


def ed8_checks(target: Path, parent: Path) -> dict[str, bool]:
    parity_directories = (
        "exclusion_audit",
        "layout_authority",
        "palette_authority",
        "plot_ready",
        "source_authority",
        "source_data",
        "statistics",
    )
    parity = all(same_tree(target, parent, relative) for relative in parity_directories)
    unchanged: list[bool] = []
    for role in "abcdefgh":
        for suffix in ("png", "tiff"):
            source = find_one(target / "individual_panels", f"ed08_panel_{role}_*.{suffix}")
            peer = parent / "individual_panels" / source.name
            unchanged.append(sha256(source) == sha256(peer))

    i_pdf = find_one(target / "individual_panels", "ed08_panel_i_*.pdf")
    ticks = [
        span
        for span in pdf_spans(i_pdf)
        if span["text"] in {"Rewarded", "Timeout"}
    ]
    horizontal = len(ticks) == 4 and all(
        close(float(span["direction"][0]), 1.0, 1e-6)
        and close(float(span["direction"][1]), 0.0, 1e-6)
        for span in ticks
    )
    manifest = json.loads(
        (target / "ED08_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    root_hashes = manifest.get("root_output_sha256", {})
    export_hashes = {
        "svg": sha256(target / "ed08_version_record.svg"),
        "pdf": sha256(target / "ed08_version_record.pdf"),
        "png_600dpi": sha256(target / "ed08_version_record_600dpi.png"),
        "tiff_600dpi": sha256(target / "ed08_version_record_600dpi.tiff"),
    }
    return {
        "ed8_scientific_authority_trees_exact": parity,
        "ed8_panels_a_to_h_rasters_exact": all(unchanged),
        "ed8_panel_i_ticks_horizontal": horizontal,
        "ed8_manifest_current_round10": manifest.get("source_round") == 10
        and manifest.get("record") == target.name,
        "ed8_manifest_root_hashes_exact": root_hashes == export_hashes,
    }


def pointer_and_caption_checks(targets: dict[str, Path]) -> dict[str, bool]:
    current_main = (RECORD_ROOT / "CURRENT_MAIN_FIGURES.md").read_text(encoding="utf-8")
    current_ed = (RECORD_ROOT / "CURRENT_EXTENDED_DATA_FIGURES.md").read_text(
        encoding="utf-8"
    )
    evidence = (CAPTION_ROOT / "CAPTION_EVIDENCE_AUDIT.md").read_text(encoding="utf-8")
    captions = (CAPTION_ROOT / "figure_captions_main1-5_extended1-10.tex").read_text(
        encoding="utf-8"
    )
    ledger = (CAPTION_ROOT / "TERMINOLOGY_LEDGER.md").read_text(encoding="utf-8")
    main_ok = all(targets[key].name in current_main for key in ("main3", "main4"))
    ed_ok = all(targets[key].name in current_ed for key in ("ed1", "ed2", "ed6", "ed7", "ed8"))
    evidence_ok = all(path.name in evidence for path in targets.values())
    combined = captions + "\n" + ledger
    return {
        "current_main_pointer_targets_exact": main_ok,
        "current_extended_pointer_targets_exact": ed_ok,
        "caption_evidence_targets_exact": evidence_ok,
        "caption_open_compound_policy": not any(
            token.lower() in combined.lower() for token in FORBIDDEN_VISIBLE
        ),
        "caption_required_open_terms_present": all(
            token in captions
            for token in ("maze tile", "long bout", "long gap", "reward directed")
        ),
    }


def visible_term_checks(targets: Iterable[Path]) -> dict[str, bool]:
    svg_text = "\n".join(
        find_one(record, "*_version_record.svg").read_text(encoding="utf-8")
        for record in targets
    )
    pdf_visible = "\n".join(
        pdf_text(find_one(record, "*_version_record.pdf")) for record in targets
    )
    return {
        "root_svg_no_forbidden_compounds": not any(
            token.lower() in svg_text.lower() for token in FORBIDDEN_VISIBLE
        ),
        "root_pdf_no_forbidden_compounds": not any(
            token.lower() in pdf_visible.lower() for token in FORBIDDEN_VISIBLE
        ),
    }


def validate() -> dict[str, Any]:
    targets = {key: resolve_record(pair[0]) for key, pair in RECORDS.items()}
    parents = {key: resolve_record(pair[1]) for key, pair in RECORDS.items()}
    checks: dict[str, bool] = {}
    expected_images = {
        "main3": 4,
        "main4": 14,
        "ed1": 0,
        "ed2": 0,
        "ed6": 0,
        "ed7": 0,
        "ed8": 2,
    }
    for key, record in targets.items():
        checks.update(
            {
                f"{key}_{name}": value
                for name, value in record_export_checks(
                    record, expected_images=expected_images[key]
                ).items()
            }
        )
    checks.update(main3_checks(targets["main3"], parents["main3"]))
    checks.update(main4_checks(targets["main4"], parents["main4"]))
    checks.update(ed1_checks(targets["ed1"], parents["ed1"]))
    checks.update(ed2_checks(targets["ed2"], parents["ed2"]))
    checks.update(ed6_checks(targets["ed6"], parents["ed6"]))
    checks.update(ed7_checks(targets["ed7"], parents["ed7"]))
    checks.update(ed8_checks(targets["ed8"], parents["ed8"]))
    checks.update(pointer_and_caption_checks(targets))
    checks.update(visible_term_checks(targets.values()))
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": "cross-figure-language-refinement-validation/v1",
        "status": "PASS" if not failures else "FAIL",
        "records": {
            key: target.relative_to(REPO).as_posix() for key, target in targets.items()
        },
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": sum(checks.values()),
        "n_fail": len(failures),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
