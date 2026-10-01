"""Build the annotation-ready PowerPoint for Extended Data Figure 6.

One slide, one full-bleed image at the figure's own canvas size, source notes in the
speaker notes. Every round writes round-suffixed files into the same figure folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from PIL import Image
import win32com.client

REPO = Path(__file__).resolve().parents[2]
FIGURE_ROOT = (
    REPO
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed06_within_session_convergence"
)
ASSEMBLY = FIGURE_ROOT / "assembly"
STEM = "ed06_within_session_convergence_assembly"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    rnd = int(args.round)

    figure = ASSEMBLY / f"{STEM}_round{rnd}.png"
    figure_tiff = ASSEMBLY / f"{STEM}_round{rnd}.tiff"
    manifest_path = ASSEMBLY / f"ED06_ASSEMBLY_MANIFEST_ROUND{rnd}.json"
    validation_path = ASSEMBLY / f"ED06_VALIDATION_ROUND{rnd}.csv"
    statistics_path = ASSEMBLY / f"ED06_DISPLAYED_STATISTICS_ROUND{rnd}.csv"
    caption_path = FIGURE_ROOT / f"CAPTION_NOTES_ROUND{rnd}.md"

    output = FIGURE_ROOT / f"extended_data_figure6_layout_review_round{rnd}.pptx"
    view_png = FIGURE_ROOT / f"extended_data_figure6_source_rerender_round{rnd}.png"
    view_tiff = FIGURE_ROOT / f"extended_data_figure6_source_rerender_round{rnd}.tiff"
    preview = Path(__file__).resolve().parent / f"extended_data_figure6_round{rnd}_review.png"
    deck_manifest_path = FIGURE_ROOT / f"POWERPOINT_MANIFEST_ROUND{rnd}.json"

    for path in (figure, manifest_path, validation_path, caption_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise AssertionError("Figure validation must pass before review-deck export")

    shutil.copy2(figure, view_png)
    if figure_tiff.is_file():
        shutil.copy2(figure_tiff, view_tiff)
    else:
        with Image.open(figure) as image:
            image.save(view_tiff, compression="tiff_lzw")

    width_in, height_in = manifest["canvas_inches"]
    width_pt = float(width_in) * 72.0
    height_pt = float(height_in) * 72.0

    gate = manifest["gate_impact_on_displayed_columns"]
    powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
    powerpoint.Visible = True
    presentation = powerpoint.Presentations.Add()
    try:
        presentation.PageSetup.SlideWidth = width_pt
        presentation.PageSetup.SlideHeight = height_pt
        slide = presentation.Slides.Add(1, 12)
        picture = slide.Shapes.AddPicture(
            str(figure), False, True, 0.0, 0.0, width_pt, height_pt
        )
        picture.Name = f"Extended Data Figure 6 source rerender round {rnd}"
        picture.AlternativeText = (
            "Extended Data Figure 6 rebuilt from the five-minute coverage-gate plot-ready "
            "tables: two method schematics plus three metric rows of fixed Day2-template "
            "convergence and sliding-pivot local stability."
        )
        notes = "\n".join(
            [
                f"Extended Data Figure 6 round-{rnd} source rerender",
                "",
                "- Layout authority: slide 7 of sup-figs-v2.pptx (merged numbering; slide 8 is",
                "  the empty 'combine with 7' note). Sources remain under old ED7.",
                "- Data authority: docs/coverage5min_window_reanalysis_2026-08-05 — each window,",
                "  and each side of a pivot, requires a fixed 300 s of selected data.",
                f"- Gate impact: 10-min unchanged; 20-min changed "
                f"({gate['20-min window']['newly_admitted_curve_rows']} newly admitted curve "
                f"rows, max |dmean| {gate['20-min window']['max_abs_mean_change']:.3f}); "
                "3,000-tile unchanged.",
                "- Panels a, b are the requested method schematics (slide note: 'add a",
                "  schematic?'), adapted from analysis_schematic.py as Main Figure 3 adapted it.",
                "- Panels c/e/g: fixed Day2-template convergence, 10-min and 20-min maze-time",
                "  plus the 3,000-tile window. Panels d/f/h: local stability across a sliding",
                "  30-min pivot, Day 1 and Day 2 on a shared range.",
                "- Every panel is drawn from plot-ready tables; no maintained panel image is",
                "  embedded. Exact q values are in the caption notes, not on the panels.",
                "",
                "[Sources]",
                str(manifest_path),
                str(validation_path),
                str(statistics_path),
                str(caption_path),
                "[/Sources]",
            ]
        )
        slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes
        presentation.SaveAs(str(output))
        slide.Export(
            str(preview), "PNG", int(round(width_pt * 4.0)), int(round(height_pt * 4.0))
        )
        time.sleep(1.0)
    finally:
        closed = False
        for _ in range(3):
            try:
                presentation.Close()
                closed = True
                break
            except Exception:
                time.sleep(1.0)
        if not closed:
            raise RuntimeError("Failed to close PowerPoint presentation")
        try:
            powerpoint.Quit()
        except Exception:
            pass

    shapes = 1
    deck_manifest = {
        "round": rnd,
        "canvas_inches": [width_in, height_in],
        "slide_count": 1,
        "picture_count": shapes,
        "figure_png": str(figure.relative_to(REPO)),
        "figure_png_sha256": sha256(figure),
        "viewable_png": str(view_png.relative_to(REPO)),
        "viewable_png_sha256": sha256(view_png),
        "viewable_tiff": str(view_tiff.relative_to(REPO)),
        "viewable_tiff_sha256": sha256(view_tiff),
        "pptx": str(output.relative_to(REPO)),
        "pptx_sha256": sha256(output),
        "preview_png": str(preview.relative_to(REPO)),
        "preview_exists": preview.is_file() and preview.stat().st_size > 0,
        "status": "pass",
    }
    deck_manifest_path.write_text(json.dumps(deck_manifest, indent=2), encoding="utf-8")
    print(json.dumps(deck_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
