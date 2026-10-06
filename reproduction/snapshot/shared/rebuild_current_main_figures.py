"""Shared rendering adapters for the publication's main figures.

The root reproduce.py stages verified inputs and calls these helpers. Figure 2
and Figure 3 use their dedicated adapters; Figures 1, 4 and 5 reuse only the
import, composition and export helpers needed by their own builders. No finished
figure is an input. See docs/CODE.md for the supported reading/entrypoint map.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types
from typing import Any, Callable
import xml.etree.ElementTree as ET

# Dynamic imports must never write cache files into immutable selected records.
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import fitz
import pandas as pd
from PIL import Image


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]


def _load_module(name: str, path: Path, *, synthetic_file: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(synthetic_file or path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    # A few historically frozen helpers retain a UTF-8 BOM.  ``utf-8-sig``
    # consumes it while remaining identical to UTF-8 for every other source.
    source = path.read_text(encoding="utf-8-sig")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _runtime_root() -> Path:
    # The public launcher authenticates each ordinary Python source before
    # staging it here. All figures use this single, browsable support tree.
    for parent in (VOR_ROOT, *VOR_ROOT.parents):
        root = parent / "shared/runtime"
        if (root / "ca_analysis/__init__.py").is_file():
            return root
    raise RuntimeError("Run this figure through reproduce.py to stage its shared runtime")


def _remove_live_source_path() -> None:
    live_src = (REPO_ROOT / "src").resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == live_src:
                continue
        except (OSError, RuntimeError):
            pass
        retained.append(entry)
    sys.path[:] = retained


def _assert_isolated_ca_analysis(record: Path, runtime_root: Path) -> None:
    allowed_roots = (runtime_root.resolve(), (record / "source_code").resolve())
    live_src = (REPO_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "ca_analysis" and not name.startswith("ca_analysis."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        resolved = Path(module_file).resolve()
        if resolved == live_src or live_src in resolved.parents:
            raise RuntimeError(f"Live repository dependency imported during source rebuild: {resolved}")
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Unfrozen ca_analysis dependency imported: {resolved}")


def _install_frozen_panel_modules(record: Path):
    runtime_root = _runtime_root()
    _remove_live_source_path()
    runtime_text = str(runtime_root)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_text]
    sys.path.insert(0, runtime_text)
    for name in tuple(sys.modules):
        if name == "ca_analysis" or name.startswith("ca_analysis."):
            del sys.modules[name]
    ca_analysis = importlib.import_module("ca_analysis")

    # Prefer the figure-local heatmap helper when one is included.
    for stem in ("level_heatmap_utils",):
        frozen = record / "source_code" / f"{stem}.py"
        if frozen.is_file():
            loaded = _load_module(f"ca_analysis.{stem}", frozen)
            setattr(ca_analysis, stem, loaded)
    frozen_panels = _load_module(
        "ca_analysis.figure_panels",
        record / "source_code" / "figure_panels.py",
    )
    setattr(ca_analysis, "figure_panels", frozen_panels)
    _assert_isolated_ca_analysis(record, runtime_root)
    return frozen_panels


def _svg_dimensions_mm(svg_path: Path) -> tuple[float, float]:
    root = ET.parse(svg_path).getroot()

    def to_mm(value: str) -> float:
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|in|pt|px)?\s*",
            str(value),
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"Unsupported SVG length: {value!r}")
        number = float(match.group(1))
        unit = (match.group(2) or "px").lower()
        return number * {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0, "px": 25.4 / 96.0}[unit]

    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width and height:
        return to_mm(width), to_mm(height)
    viewbox = [float(value) for value in root.attrib["viewBox"].split()]
    return viewbox[2] * 25.4 / 72.0, viewbox[3] * 25.4 / 72.0


def _find_chrome() -> Path:
    from ca_analysis.browser_export import find_browser

    return find_browser(preferred=("chrome", "edge", "chromium"))


def _wait_for_file(path: Path, *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable = 0
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.is_file() else -1
        if size > 0 and size == previous_size:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        previous_size = size
        time.sleep(0.15)
    raise RuntimeError(f"Export did not settle: {path}")


def _export_svg_bundle(source_svg: Path, output_stem: Path) -> list[Path]:
    """Materialize the accepted editable/raster output contract from fresh SVG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    target_svg = output_stem.with_suffix(".svg")
    if source_svg.resolve() != target_svg.resolve():
        shutil.copy2(source_svg, target_svg)
    width_mm, height_mm = _svg_dimensions_mm(target_svg)
    html = output_stem.with_suffix(".html")
    svg_text = target_svg.read_text(encoding="utf-8")
    html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"@page{{size:{width_mm:.6f}mm {height_mm:.6f}mm;margin:0}}"
        f"html,body{{margin:0;width:{width_mm:.6f}mm;height:{height_mm:.6f}mm;overflow:hidden}}"
        f"svg{{display:block;width:{width_mm:.6f}mm!important;height:{height_mm:.6f}mm!important}}"
        f"</style></head><body>{svg_text}</body></html>",
        encoding="utf-8",
    )
    pdf = output_stem.with_suffix(".pdf")
    command = [
        str(_find_chrome()),
        "--headless=new",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf}",
        html.resolve().as_uri(),
    ]
    from ca_analysis.browser_export import platform_headless_flags, run_browser
    command[1:1] = platform_headless_flags()
    result = run_browser(command, check=False, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            "Browser SVG-to-PDF export failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    _wait_for_file(pdf)
    png = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".png")
    tiff = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".tiff")
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page figure PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    image.save(png, dpi=(600, 600), optimize=True)
    image.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    return [target_svg, pdf, png, tiff, html]


def _load_publication_renderer(record: Path, name: str):
    _install_frozen_panel_modules(record)
    return _load_module(
        name,
        record / "source_code" / "render_main_figs_v3_publication.py",
        synthetic_file=REPO_ROOT / "scripts" / "render_main_figs_v3_publication.py",
    )


def _render_main2(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    publication = _load_publication_renderer(record, "record_local_main2_publication")
    distribution = record / "source_data" / "distribution"
    consecutive = record / "source_data" / "consecutive_gap"
    gap = record / "source_data" / "gap_curve"
    rate = record / "source_data" / "rate_chain"
    publication.FIGURE2_HELPER = record / "source_code" / "preview_figure2_assembly.py"
    publication.EXAMPLE_SAVED_DIR = record / "panel_b_sources"
    publication.FIGURE2_DISTRIBUTION_DIR = distribution
    publication.FIGURE2_GAP_CURVE = gap / "gap_with_day1_day2_curve_data.csv"
    publication.FIGURE2_CONSECUTIVE_POINTS = (
        consecutive / "consecutive_gap_clean_long_gap_corrected_points.csv"
    )
    publication.FIGURE2_CONSECUTIVE_STATS = (
        consecutive / "consecutive_gap_clean_long_gap_corrected_stats.csv"
    )
    publication.FIGURE2_RATE_CHAIN_POINTS = (
        rate / "session_pair_rate_consecutive_gap_points_canonical.csv"
    )
    publication.FIGURE2_RATE_CHAIN_STATS = (
        rate / "session_pair_rate_consecutive_gap_stats_canonical.csv"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    style = publication.PublicationStyle(dpi=300)
    manifest = publication.render_figure2(output_dir, style)
    manifest_path = output_dir / "figure2_source_manifest.json"
    return [
        *_export_svg_bundle(
            output_dir / "figure2_source_publication.svg",
            output_dir / "main_figure_2_source_rebuild",
        ),
        manifest_path,
    ]


def _render_main3(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    publication = _load_publication_renderer(record, "record_local_main3_publication")
    first30 = record / "source_data" / "first30"
    long_gap = record / "source_data" / "long_gap"
    publication.EXAMPLE_SAVED_DIR = record / "source_data" / "example"
    publication.DAY2_TEMPLATE_GROUP_CURVES = first30 / "day2_template_group_curve_summary.csv"
    publication.FIGURE3_FIRST30_POINTS_ALL_METRICS = (
        first30 / "combined_first30_points_all_metrics.csv"
    )
    publication.FIGURE3_FIRST30_STATS_ALL_METRICS = (
        first30 / "combined_first30_stats_all_metrics.csv"
    )
    publication.LONG_GAP_POINTS = (
        long_gap / "main3_experience_trend_points_no_manual_mask.csv"
    )
    publication.LONG_GAP_STATS = (
        long_gap / "main3_experience_trend_clustered_stats_no_manual_mask.csv"
    )
    publication.FIGURE3_CURVE_SUPPORT_MASK = (
        record / "source_data" / "support" / "ed07_current59_curve_support_mask.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    style = publication.PublicationStyle(dpi=600)
    publication.render_figure3(output_dir, style)
    # Preserve the source renderer's exact page dimensions and 600-dpi images.
    # Re-exporting the SVG's rounded millimetre attributes changes edge pixels.
    output_stem = output_dir / "main_figure_3_source_rebuild"
    outputs = []
    for suffix in (".svg", ".pdf", ".html"):
        target = output_stem.with_suffix(suffix)
        shutil.copy2(output_dir / f"figure3_source_publication{suffix}", target)
        outputs.append(target)
    with fitz.open(output_stem.with_suffix(".pdf")) as document:
        if document.page_count != 1:
            raise RuntimeError("Expected one-page Figure 3 source-rendered PDF")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    png = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".png")
    tiff = png.with_suffix(".tiff")
    image.save(png, dpi=(600, 600), optimize=True)
    image.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    return [
        *outputs,
        png,
        tiff,
        output_dir / "figure3_source_manifest.json",
    ]


def _compose_main4(publication, module, payloads: dict[str, bytes], output_svg: Path) -> None:
    width_pt = module.FIGSIZE[0] * 72.0
    height_pt = module.FIGSIZE[1] * 72.0
    root = ET.Element(
        f"{{{module.SVG_NS}}}svg",
        {
            "width": f"{module.WIDTH_MM:.3f}mm",
            "height": f"{module.HEIGHT_MM:.3f}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
            "version": "1.1",
        },
    )
    # Panel a uses the accepted PPT-style source crop rather than aspect-fit
    # placement.  Creating defs before the white background also preserves the
    # accepted editable SVG structure.
    ET.SubElement(root, f"{{{module.SVG_NS}}}defs")
    ET.SubElement(
        root,
        f"{{{module.SVG_NS}}}rect",
        {"width": "100%", "height": "100%", "fill": "white"},
    )
    for panel in "abcdef":
        if panel == "a":
            source = ET.fromstring(payloads[panel])
            _x0, _y0, source_w, source_h = module.svg_viewbox(source)
            publication._add_svg_panel_cropped_absolute(
                root,
                {
                    "children": [copy.deepcopy(child) for child in list(source)],
                    "viewbox_width": source_w,
                    "viewbox_height": source_h,
                },
                panel_id="figure4_a",
                slot_pt=tuple(value * 72.0 for value in module.SLOTS[panel]),
                src_rect=publication.FIGURE4_PANEL_A_PPT_CROP,
            )
        else:
            module.add_svg_panel(
                root,
                payloads[panel],
                panel_id=f"figure4_{panel}",
                slot=module.SLOTS[panel],
            )
    for panel, (x, y) in module.LABEL_POSITIONS.items():
        node = ET.SubElement(
            root,
            f"{{{module.SVG_NS}}}text",
            {
                "x": f"{x * 72.0:.4f}",
                "y": f"{y * 72.0:.4f}",
                "font-family": "Arial",
                "font-size": "10.5",
                "font-weight": "bold",
                "fill": "black",
            },
        )
        node.text = panel
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)


RENDERERS: dict[str, Callable[[Path, Path, dict[str, Any]], list[Path]]] = {
    "main_figure_2": _render_main2,
}
