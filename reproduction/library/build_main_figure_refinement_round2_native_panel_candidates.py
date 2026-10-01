"""Rerender changed Main Figure 4/5 panels natively and reassemble candidates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import render_main_figs_v3_publication as publication  # noqa: E402


FIGURE4_PARENT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure4_2026-08-11_round4_complete_upright_i_density_ticks"
    / "figure4_version_record.svg"
)
FIGURE5_PARENT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure5_2026-08-08_full_audit"
    / "figure5_version_record.svg"
)


def _export_bundle(
    svg_path: Path,
    *,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> dict[str, str]:
    png_path = svg_path.with_suffix(".png")
    pdf_path = svg_path.with_suffix(".pdf")
    tiff_path = svg_path.with_suffix(".tiff")
    html_path = publication._export_svg_with_chrome(
        svg_path,
        png_path,
        pdf_path,
        width_mm=width_mm,
        height_mm=height_mm,
        dpi=dpi,
    )
    with Image.open(png_path) as image:
        image.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    return {
        key: str(path.relative_to(REPO_ROOT))
        for key, path in {
            "svg": svg_path,
            "png": png_path,
            "pdf": pdf_path,
            "tiff": tiff_path,
            "html": html_path,
        }.items()
    }


def _find_direct_child(root: ET.Element, element_id: str) -> tuple[int, ET.Element]:
    for index, child in enumerate(list(root)):
        if child.attrib.get("id") == element_id:
            return index, child
    raise RuntimeError(f"Missing direct child {element_id!r}")


def build_figure4(out_dir: Path, *, dpi: int) -> dict[str, object]:
    style = publication.PublicationStyle(dpi=dpi)
    with publication.plt.rc_context(style.rcparams()):
        figure, source = publication._render_figure4_method_schematic(
            (style.figure4_panel_a_slot[2], style.figure4_panel_a_slot[3]),
            style,
        )
        payload = publication._figure_to_svg_payload(
            "figure4_a",
            figure,
            dpi=dpi,
            fixed_bbox=True,
        )
    panel_dir = out_dir / "figure4_individual_panels"
    panel_exports = publication._write_svg_payload_outputs(
        payload,
        panel_dir,
        "figure4_panel_a_decoding_method_schematic",
        dpi=dpi,
        facecolor=style.facecolor,
    )

    root = ET.parse(FIGURE4_PARENT).getroot()
    old_index, _old_group = _find_direct_child(root, "panel_figure4_a")
    root.remove(_old_group)
    publication._add_svg_panel_cropped_absolute(
        root,
        payload,
        panel_id="figure4_a",
        slot_pt=tuple(float(value) * 72.0 for value in style.figure4_panel_a_slot),
        src_rect=publication.FIGURE4_PANEL_A_PPT_CROP,
    )
    new_group = list(root)[-1]
    root.remove(new_group)
    root.insert(old_index, new_group)

    svg_path = out_dir / "figure4_source_publication.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    outputs = _export_bundle(svg_path, width_mm=180.0, height_mm=96.0, dpi=dpi)
    return {
        "parent_svg": str(FIGURE4_PARENT.relative_to(REPO_ROOT)),
        "native_renderer": "render_main_figs_v3_publication._render_figure4_method_schematic",
        "composite_assembler": "render_main_figs_v3_publication._add_svg_panel_cropped_absolute",
        "source": source,
        "outputs": outputs,
        "individual_panels": {
            "a": {
                key: str(value.relative_to(REPO_ROOT)) if isinstance(value, Path) else value
                for key, value in panel_exports.items()
            }
        },
    }


def build_figure5(out_dir: Path, *, dpi: int) -> dict[str, object]:
    helper = publication._load_script_module(
        "preview_figure3_assembly_native_candidate",
        publication.FIGURE5_HELPER,
    )
    panel_spec = helper.PANEL_LAYOUT_SPEC["e"]
    figure = helper._render_panel_e(
        (float(panel_spec["render_w"]), float(panel_spec["render_h"]))
    )
    payload = publication._figure_to_svg_payload(
        "figure5_e",
        figure,
        dpi=dpi,
        fixed_bbox=True,
    )
    panel_dir = out_dir / "figure5_individual_panels"
    panel_exports = publication._write_svg_payload_outputs(
        payload,
        panel_dir,
        "figure5_panel_b_level_code_heatmap",
        dpi=dpi,
    )

    root = ET.parse(FIGURE5_PARENT).getroot()
    _index, group = _find_direct_child(root, "panel_e")
    for child in list(group):
        group.remove(child)
    for child in payload["children"]:
        group.append(copy.deepcopy(child))

    svg_path = out_dir / "figure5_source_publication.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    outputs = _export_bundle(
        svg_path,
        width_mm=180.0,
        height_mm=124.23121387283236,
        dpi=dpi,
    )
    return {
        "parent_svg": str(FIGURE5_PARENT.relative_to(REPO_ROOT)),
        "native_renderer": "preview_figure3_assembly._render_panel_e",
        "composite_assembler": "preview_figure3_assembly panel_e slot-preserving replacement",
        "source_validation": helper.validate_figure5_corrected842_sources(),
        "outputs": outputs,
        "individual_panels": {
            "b": {
                key: str(value.relative_to(REPO_ROOT)) if isinstance(value, Path) else value
                for key, value in panel_exports.items()
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "figure_refinement_round2_2026-08-11"
            / "main_figures"
            / "native_figure4_5"
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--figures", nargs="+", choices=("4", "5"), default=("4", "5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    if "4" in args.figures:
        report["figure4"] = build_figure4(args.out_dir, dpi=args.dpi)
    if "5" in args.figures:
        report["figure5"] = build_figure5(args.out_dir, dpi=args.dpi)
    publication._write_json(args.out_dir / "native_candidate_build.json", report)
    print(report)


if __name__ == "__main__":
    main()
