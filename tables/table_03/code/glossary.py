"""Build Supplementary Table 3 from its editable glossary source."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def markup(text: str) -> str:
    """Keep source definitions plain text; typeset the one recurring subscript."""
    return escape(text).replace("N32", "<i>N</i><sub>32</sub>")


def load_glossary(source: Path) -> dict:
    """Read the terms and definitions in their publication order."""
    return json.loads(source.read_text(encoding="utf-8"))


def render_glossary(glossary: dict, output: Path) -> Path:
    """Save the two-page glossary with the publication layout."""
    pages = glossary["pages"]
    if len(pages) != 2:
        raise ValueError("This table layout expects one behavioral and one neural page")
    terms = [entry["term"] for page in pages for entry in page["entries"]]
    if len(terms) != len(set(terms)):
        raise ValueError("Duplicate glossary terms")
    for page in pages:
        for entry in page["entries"]:
            if not entry["term"].strip() or not entry["meaning"].strip():
                raise ValueError("Empty glossary term or definition")

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title=glossary["title"],
        author="",
        subject="Definitions of behavioral and neural terms",
        invariant=1,
    )
    title = ParagraphStyle(
        "Table title", fontName="Helvetica-Bold", fontSize=12, leading=15,
        spaceAfter=7,
    )
    section = ParagraphStyle(
        "Table section", fontName="Helvetica-Bold", fontSize=10, leading=13,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "Definition", fontName="Helvetica", fontSize=9.5, leading=12.4,
    )
    term = ParagraphStyle(
        "Term", parent=body, fontName="Helvetica-Bold",
    )
    header = ParagraphStyle(
        "Column header", parent=term, fontSize=9.5,
    )
    story = []
    for page_number, page in enumerate(pages, 1):
        if page_number > 1:
            story.append(PageBreak())
        heading = glossary["title"] + (" (continued)" if page_number > 1 else "")
        story.extend([
            Paragraph(escape(heading), title),
            Paragraph(escape(page["heading"]), section),
        ])
        rows = [[Paragraph("Term", header), Paragraph("Meaning", header)]]
        rows.extend([
            [Paragraph(markup(e["term"]), term), Paragraph(markup(e["meaning"]), body)]
            for e in page["entries"]
        ])
        table = Table(rows, colWidths=[49 * mm, 127 * mm], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4E4E4")),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F2F2F2")),
            ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.HexColor("#666666")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.HexColor("#A0A0A0")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#C8C8C8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Spacer(1, 3 * mm))

    def footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(A4[0] - 17 * mm, 9 * mm, f"{doc.page} of {len(pages)}")
        canvas.restoreState()

    document.build(
        story, onFirstPage=footer, onLaterPages=footer,
    )
    return output
