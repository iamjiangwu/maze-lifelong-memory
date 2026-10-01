"""Render Supplementary Table 2 as a compact mouse-by-return-month matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from datetime import date

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from PIL import Image


HEADER_FILL = "#E6E6E6"
MOUSE_FILL = "#F2F2F2"
TEXT = "#202020"
RULE = "#C8C8C8"
STRONG_RULE = "#666666"
WIDTH_MM = 297.0
HEIGHT_MM = 210.0
DPI = 600
RETURN_MONTHS = ("3", "6", "9", "12")
EXCLUDED_PAIR = "1616day1->1616bday1"
EXCLUDED_180_PAIR = "618-6day1->618-12day1"
RETAINED_1616_PAIR = "1616bday1->1616b-3day1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_source(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 58:
        raise ValueError(f"Expected 58 comprehensive rows, found {len(rows)}")
    if len({row["Animal ID"] for row in rows}) != 18:
        raise ValueError("Expected 18 unique animals")
    if sum(parse_bool(row["Fixed-month behavior return roster"]) for row in rows) != 56:
        raise ValueError("Expected 56 behavior return intervals")
    if sum(parse_bool(row["Maintained calcium imaging"]) for row in rows) != 37:
        raise ValueError("Expected 37 calcium-imaged intervals (36 long-gap and one short)")
    if sum(row["Pair key"] == EXCLUDED_PAIR for row in rows) != 1:
        raise ValueError("Expected exactly one displayed short mouse-1616 pair")
    if sum(row["Pair key"] == RETAINED_1616_PAIR and row["Return month"] == "6" for row in rows) != 1:
        raise ValueError("The retained mouse-1616 interval is not assigned to 6 months")
    if sum(row["Pair key"] == EXCLUDED_180_PAIR and row["Return month"] == "12" for row in rows) != 1:
        raise ValueError("The mouse-618 180-d interval is not retained at 12 months")
    if not any(
        row["Animal ID"] == "418"
        and row["Return month"] == "12"
        and parse_bool(row["Fixed-month behavior return roster"])
        for row in rows
    ):
        raise ValueError("Mouse 418's 12-month behavior interval is missing")
    for row in rows:
        expected = (date.fromisoformat(row["Return Day i date"]) - date.fromisoformat(row["Preceding Day i+1 date"])).days
        if row["Interval (d)"] != str(expected):
            raise ValueError(f"Non-calendar-day interval: {row['Pair key']}")
    return rows


def natural_month_list(months: list[str]) -> str:
    if not months:
        return "None"
    return ", ".join(months)


def build_matrix(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    matrix: list[dict[str, object]] = []
    for animal in sorted({row["Animal ID"] for row in rows}, key=int):
        animal_rows = [row for row in rows if row["Animal ID"] == animal]
        output: dict[str, object] = {
            "animal": animal,
            "3": None,
            "6": None,
            "9": None,
            "12": None,
            "additional": [],
            "calcium": "None",
        }
        calcium_months: list[str] = []
        additional_calcium = False
        for row in animal_rows:
            interval = int(row["Interval (d)"])
            month = row["Return month"]
            if month in RETURN_MONTHS:
                if output[month] is not None:
                    raise ValueError(f"Duplicate return month for mouse {animal}: {month}")
                output[month] = interval
                if parse_bool(row["Maintained calcium imaging"]):
                    calcium_months.append(month)
            else:
                output["additional"].append(interval)
                additional_calcium = parse_bool(row["Maintained calcium imaging"])
        if calcium_months:
            output["calcium"] = natural_month_list(sorted(calcium_months, key=RETURN_MONTHS.index))
            if additional_calcium:
                output["calcium"] += "; other interval"
        elif additional_calcium:
            output["calcium"] = "Other intervals"
        matrix.append(output)
    return matrix


def add_rect(ax, x: float, y: float, width: float, height: float, color: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    weight: str = "normal",
    style: str = "normal",
    ha: str = "left",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        color=TEXT,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        clip_on=False,
    )


def draw_table(matrix: list[dict[str, object]], source_rows: list[dict[str, str]]) -> plt.Figure:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.025, 0.025
    full_width = 1.0 - left - right
    add_text(
        ax,
        left,
        0.950,
        "Supplementary Table 2. Intervals between successive recording cycles",
        size=12.0,
        weight="bold",
    )

    table_top, table_bottom = 0.895, 0.30
    header_height = 0.064
    row_height = (table_top - table_bottom - header_height) / len(matrix)
    column_fracs = [0.08, 0.115, 0.115, 0.115, 0.115, 0.13, 0.33]
    headers = [
        "Mouse",
        "3-month return\n(d)",
        "6-month return\n(d)",
        "9-month return\n(d)",
        "12-month return\n(d)",
        "Other intervals\n(d)",
        "Calcium-imaged returns\n(months)",
    ]
    boundaries = [left]
    for fraction in column_fracs:
        boundaries.append(boundaries[-1] + full_width * fraction)

    add_rect(ax, left, table_top - header_height, full_width, header_height, HEADER_FILL)
    for index, header in enumerate(headers):
        add_text(
            ax,
            (boundaries[index] + boundaries[index + 1]) / 2.0,
            table_top - header_height / 2.0,
            header,
            size=6.4,
            weight="bold",
            ha="center",
        )

    for row_index, row in enumerate(matrix):
        y_top = table_top - header_height - row_index * row_height
        y_bottom = y_top - row_height
        y_center = (y_top + y_bottom) / 2.0
        add_rect(ax, boundaries[0], y_bottom, boundaries[1] - boundaries[0], row_height, MOUSE_FILL)
        add_text(
            ax,
            (boundaries[0] + boundaries[1]) / 2.0,
            y_center,
            str(row["animal"]),
            size=6.8,
            weight="bold",
            ha="center",
        )
        for offset, key in enumerate((*RETURN_MONTHS, "additional"), start=1):
            value = row[key]
            add_text(
                ax,
                (boundaries[offset] + boundaries[offset + 1]) / 2.0,
                y_center,
                ", ".join(map(str, value)) if isinstance(value, list) else "" if value is None else str(int(value)),
                size=6.8,
                ha="center",
            )
        add_text(
            ax,
            boundaries[6] + 0.010 * full_width,
            y_center,
            str(row["calcium"]),
            size=6.6,
        )
        ax.plot(
            [left, left + full_width],
            [y_bottom, y_bottom],
            color=RULE,
            linewidth=0.32,
            transform=ax.transAxes,
            clip_on=False,
        )

    for boundary in boundaries:
        outer = boundary in (boundaries[0], boundaries[-1])
        ax.plot(
            [boundary, boundary],
            [table_bottom, table_top],
            color=STRONG_RULE if outer else RULE,
            linewidth=0.55 if outer else 0.32,
            transform=ax.transAxes,
            clip_on=False,
        )
    for y in (table_top, table_bottom):
        ax.plot(
            [left, left + full_width],
            [y, y],
            color=STRONG_RULE,
            linewidth=0.65,
            transform=ax.transAxes,
            clip_on=False,
        )

    behavior_rows = [row for row in source_rows if parse_bool(row["Fixed-month behavior return roster"])]
    month_rows = [row for row in source_rows if row["Return month"] in RETURN_MONTHS]
    approximately_three_month_rows = [
        row
        for row in source_rows
        if row["Pair key"] not in {EXCLUDED_PAIR, EXCLUDED_180_PAIR}
    ]
    approximately_three_month_intervals = [
        float(row["Interval (d)"]) for row in approximately_three_month_rows
    ]
    month_calcium = sum(parse_bool(row["Maintained calcium imaging"]) for row in month_rows)
    notes = json.loads((Path(__file__).resolve().parents[1] / 'source_data/publication_notes.json').read_text(encoding='utf-8'))
    assert len(month_rows) == 57 and len(behavior_rows) == 56 and month_calcium == 36
    assert len(approximately_three_month_rows) == 56
    assert round(statistics.mean(approximately_three_month_intervals)) == 90
    assert round(statistics.stdev(approximately_three_month_intervals)) == 4
    note_size = 8.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family='Arial', size=note_size)
    max_width_px = (full_width - 0.008) * fig.bbox.width
    y = 0.267
    for note in notes:
        lines = []
        line = ''
        for word in note.split():
            proposed = f'{line} {word}'.strip()
            width = renderer.get_text_width_height_descent(proposed, font, False)[0]
            if line and width > max_width_px:
                lines.append(line)
                line = word
            else:
                line = proposed
        lines.append(line)
        artist = ax.text(left + 0.004, y, '\n'.join(lines), transform=ax.transAxes,
                         fontsize=note_size, color=TEXT, va='top', linespacing=1.3)
        fig.canvas.draw()
        bounds = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        assert bounds.x1 <= 0.976 and bounds.y0 >= 0.025, bounds
        y = bounds.y0 - 0.010
    return fig


def write_outputs(
    fig: plt.Figure,
    matrix: list[dict[str, object]],
    source_rows: list[dict[str, str]],
    source_csv: Path,
    output_dir: Path,
    pdf_output: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Supplementary_Table_2_Long_Gap_Intervals_publication"
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": pdf_output,
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    fig.savefig(paths["svg"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(paths["png"], dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(
        paths["tiff"],
        dpi=DPI,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    with Image.open(paths["png"]) as image:
        png_size = list(image.size)
    with Image.open(paths["tiff"]) as image:
        tiff_size = list(image.size)

    required = [
        "Supplementary Table 2",
        "418",
        "90 ± 4 d",
        "180-d",
        "110",
        "excluded mouse 1616's 14-d interval",
        "56 intervals from 17 mice",
        "36 long-gap intervals",
        "Extended Data Fig. 2b-i",
        "Other intervals",
    ]
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text was converted to paths")
    import xml.etree.ElementTree as ET
    svg_selectable_text = re.sub(r'\s+', ' ', ' '.join(ET.fromstring(svg_text).itertext()))
    for label in required:
        if label not in svg_selectable_text:
            raise RuntimeError(f"Missing required SVG label: {label}")
    if re.search(r"\b\w*day[12]\b", svg_text, flags=re.IGNORECASE):
        raise RuntimeError("A session identifier remains in the reader-facing SVG")

    with fitz.open(paths["pdf"]) as pdf:
        if pdf.page_count != 1:
            raise RuntimeError(f"Expected one PDF page, found {pdf.page_count}")
        page = pdf[0]
        pdf_size_pt = [float(page.rect.width), float(page.rect.height)]
        expected = [WIDTH_MM / 25.4 * 72.0, HEIGHT_MM / 25.4 * 72.0]
        if any(abs(actual - target) > 0.5 for actual, target in zip(pdf_size_pt, expected)):
            raise RuntimeError(f"Unexpected PDF size: {pdf_size_pt}")
        pdf_text = re.sub(r'\s+', ' ', page.get_text("text"))
        if len(page.get_images(full=True)):
            raise RuntimeError("The publication PDF contains embedded raster images")
        for label in required:
            if label not in pdf_text:
                raise RuntimeError(f"Missing selectable PDF label: {label}")
        if re.search(r"\b\w*day[12]\b", pdf_text, flags=re.IGNORECASE):
            raise RuntimeError("A session identifier remains in the reader-facing PDF")

    behavior_rows = [row for row in source_rows if parse_bool(row["Fixed-month behavior return roster"])]
    approximately_three_month_rows = [
        row
        for row in source_rows
        if row["Pair key"] not in {EXCLUDED_PAIR, EXCLUDED_180_PAIR}
    ]
    approximately_three_month_values = [
        float(row["Interval (d)"]) for row in approximately_three_month_rows
    ]
    qa = {
        "artifact": "Publication Supplementary Table 2 interval matrix",
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256(source_csv),
        "backend": "Python matplotlib",
        "footnote_font_pt": 8.0,
        "previous_footnote_font_pt": 5.45,
        "publication_standard": "one-page A4 landscape journal table",
        "figure_size_mm": [WIDTH_MM, HEIGHT_MM],
        "pdf_page_size_pt": pdf_size_pt,
        "pdf_page_count": 1,
        "pdf_selectable_text": True,
        "pdf_vector_only": True,
        "raster_dpi": DPI,
        "png_pixels": png_size,
        "tiff_pixels": tiff_size,
        "matrix_rows": len(matrix),
        "behavior_intervals": len(behavior_rows),
        "behavior_animals": 17,
        "month_column_intervals": sum(row["Return month"] in RETURN_MONTHS for row in source_rows),
        "month_column_calcium_intervals": sum(
            row["Return month"] in RETURN_MONTHS and parse_bool(row["Maintained calcium imaging"])
            for row in source_rows
        ),
        "behavior_and_calcium_intervals": sum(
            parse_bool(row["Fixed-month behavior return roster"])
            and parse_bool(row["Maintained calcium imaging"])
            for row in source_rows
        ),
        "behavior_only_intervals": sum(
            parse_bool(row["Fixed-month behavior return roster"])
            and not parse_bool(row["Maintained calcium imaging"])
            for row in source_rows
        ),
        "maintained_calcium_intervals": sum(
            parse_bool(row["Maintained calcium imaging"]) for row in source_rows
        ),
        "approximately_3_month_interval_n": len(approximately_three_month_values),
        "approximately_3_month_interval_mean_days": statistics.mean(approximately_three_month_values),
        "approximately_3_month_interval_sample_sd_days": statistics.stdev(approximately_three_month_values),
        "approximately_3_month_interval_range_days": [
            min(approximately_three_month_values),
            max(approximately_three_month_values),
        ],
        "mouse_418_12_month_days": next(
            float(row["Interval (d)"])
            for row in source_rows
            if row["Animal ID"] == "418" and row["Return month"] == "12"
        ),
        "nonstandard_intervals": [
            {
                "animal": row["Animal ID"],
                "return_month": row["Return month"],
                "interval_days": float(row["Interval (d)"]),
                "note": row["Schedule note"],
            }
            for row in source_rows
            if row["Schedule note"]
        ],
        "session_names_in_reader_facing_table": False,
        "short_pair_excluded_from_fixed_month_and_long_gap_summaries": EXCLUDED_PAIR,
        "approximately_3_month_excluded_pairs": [EXCLUDED_PAIR, EXCLUDED_180_PAIR],
        "display_month_assignments": {RETAINED_1616_PAIR: 6, EXCLUDED_180_PAIR: 12},
        "matrix": matrix,
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    qa_path = output_dir / "Supplementary_Table_2_Long_Gap_Intervals_publication_QA.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_source(args.source_csv)
    matrix = build_matrix(source_rows)
    fig = draw_table(matrix, source_rows)
    try:
        qa = write_outputs(
            fig,
            matrix,
            source_rows,
            args.source_csv,
            args.output_dir,
            args.pdf_output,
        )
    finally:
        plt.close(fig)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
