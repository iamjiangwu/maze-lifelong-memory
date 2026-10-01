"""Read-only comparison of the S2 successor with its accepted Round 4 source."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
import openpyxl


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serializable(obj):
    return ET.tostring(obj.to_tree(), encoding="unicode")


def main():
    task = Path(__file__).resolve().parents[1]
    repo = task.parents[3]
    baseline = repo / "docs/figure_version_records/supplementary_table02_2026-09-09_round4_display_months"
    audit = task.parent / "table2_audit"
    build = task / "build"
    stem = "Supplementary_Table_2_Long_Gap_Intervals"
    accepted = baseline / "accepted_outputs"
    record = json.loads((baseline / "SOURCE_REBUILD_MANIFEST.json").read_text())
    inputs = []
    for item in record["inputs"]:
        old, new = baseline / item["path"], task / item["path"]
        assert sha256(old) == sha256(new) == item["sha256"]
        inputs.append({"path": item["path"], "sha256": sha256(new)})
    for item in record["code"] + record["outputs"]:
        assert sha256(baseline / item["path"]) == item["sha256"], item["path"]

    old_rows = list(csv.DictReader((accepted / f"{stem}.csv").open(encoding="utf-8-sig", newline="")))
    new_rows = list(csv.DictReader((build / f"{stem}.csv").open(encoding="utf-8-sig", newline="")))
    assert len(old_rows) == len(new_rows) == 58
    ledger_diffs = [{"row": i + 2, "field": k, "before": a[k], "after": b[k]}
                    for i, (a, b) in enumerate(zip(old_rows, new_rows)) for k in a if a[k] != b[k]]
    assert len(ledger_diffs) == 1
    delta = ledger_diffs[0]
    assert delta["row"] == 58 and delta["field"] == "Schedule note"
    assert delta["after"] == delta["before"].replace("ED Fig. 9l", "ED Fig. 9o")

    approved = {("Supplementary Table 2", "A26"), ("Source Data", "M58"),
                ("Summary and provenance", "C33"), ("Summary and provenance", "C34")}
    views = {}
    for data_only in [False, True]:
        old_wb = openpyxl.load_workbook(accepted / f"{stem}.xlsx", data_only=data_only)
        new_wb = openpyxl.load_workbook(build / f"{stem}.xlsx", data_only=data_only)
        assert old_wb.sheetnames == new_wb.sheetnames
        diffs, formulas, formula_changes, cells = [], 0, [], 0
        for name in old_wb.sheetnames:
            a, b = old_wb[name], new_wb[name]
            assert a.max_row == b.max_row and a.max_column == b.max_column
            assert str(a.merged_cells) == str(b.merged_cells)
            assert a.freeze_panes == b.freeze_panes
            assert a.print_area == b.print_area and a.print_title_rows == b.print_title_rows
            for key in ["sheet_format", "page_setup", "page_margins", "print_options", "sheet_view", "auto_filter", "data_validations"]:
                assert serializable(getattr(a, key)) == serializable(getattr(b, key)), (name, key)
            assert set(a.column_dimensions) == set(b.column_dimensions)
            for key in a.column_dimensions:
                assert dict(a.column_dimensions[key]) == dict(b.column_dimensions[key]), (name, "column", key)
            assert set(a.row_dimensions) == set(b.row_dimensions)
            for key in a.row_dimensions:
                assert dict(a.row_dimensions[key]) == dict(b.row_dimensions[key]), (name, "row", key)
            assert set(a.tables) == set(b.tables)
            for key in a.tables:
                assert serializable(a.tables[key]) == serializable(b.tables[key]), (name, "table", key)
            for row in a:
                for cell in row:
                    other = b[cell.coordinate]
                    cells += 1
                    if cell.value != other.value:
                        diffs.append({"sheet": name, "cell": cell.coordinate, "before": cell.value, "after": other.value})
                    if cell.data_type == "f":
                        formulas += 1
                        if cell.value != other.value:
                            formula_changes.append((name, cell.coordinate))
                    assert cell.data_type == other.data_type
                    assert cell._style == other._style, (name, cell.coordinate, "style")
                    assert cell.number_format == other.number_format
                    assert cell.hyperlink == other.hyperlink
            for key in ["_fonts", "_fills", "_borders", "_alignments", "_protections", "_number_formats", "_cell_styles"]:
                assert getattr(old_wb, key) == getattr(new_wb, key), key
        assert {(d["sheet"], d["cell"]) for d in diffs} == approved
        for d in diffs:
            if d["cell"] == "A26":
                assert d["after"] == d["before"].replace("Extended Data Fig. 9l", "Extended Data Fig. 9o")
            elif d["cell"] == "M58":
                assert d["after"] == d["before"].replace("ED Fig. 9l", "ED Fig. 9o")
            elif d["cell"] == "C33":
                assert d["after"] == sha256(build / "comprehensive_interval_ledger.csv")
            elif d["cell"] == "C34":
                assert d["after"] == sha256(build / "comprehensive_interval_ledger_manifest.json")
        assert not formula_changes
        if not data_only:
            assert formulas == 162
        views["cached_values" if data_only else "formulas"] = {
            "cells_compared": cells, "differences": diffs, "formula_count": formulas,
            "all_other_values_identical": True, "styles_and_structure_identical": True,
        }

    old_pdf, new_pdf = fitz.open(accepted / f"{stem}.pdf"), fitz.open(build / f"{stem}.pdf")
    assert len(old_pdf) == len(new_pdf) == 1
    assert old_pdf[0].rect == new_pdf[0].rect
    old_text, new_text = old_pdf[0].get_text(), new_pdf[0].get_text()
    assert old_text.count("Extended Data Fig. 9l") == 1
    assert old_text.replace("Extended Data Fig. 9l", "Extended Data Fig. 9o") == new_text

    source_changes = {}
    replacements = {
        "build_source.py": ("ED Fig. 9l", "ED Fig. 9o"),
        "build_workbook.mjs": ("Extended Data Fig. 9l", "Extended Data Fig. 9o"),
        "render_publication_table.py": ("Extended Data Fig. 9l", "Extended Data Fig. 9o"),
        "rebuild.py": ("81dc6a0d8a0981839d761076e061b24f7b38e0f977f9c00d51126185ef1681f2",
                       "b252a26476802f2e5522bec360d45f8fdb278dd425d3f2dd48ee083e51dc6ca4"),
    }
    for name, (before, after) in replacements.items():
        old = (baseline / "source_code" / name).read_text(encoding="utf-8")
        new = (task / "source_code" / name).read_text(encoding="utf-8")
        assert old.count(before) == 1
        assert old.replace(before, after) == new, name
        source_changes[name] = {"before": before, "after": after, "only_change": True}
    assert sha256(baseline / "source_code/validate_workbook.mjs") == sha256(task / "source_code/validate_workbook.mjs")
    source_build = json.loads((build / "SOURCE_REBUILD_VALIDATION.json").read_text())
    assert source_build["status"] == "pass"
    receipt = {
        "status": "PASS", "date": "2026-09-28", "artifact": "Supplementary Table 2",
        "scope": "Task-local native rebuild; two panel-reference cells and dependent provenance hashes only",
        "accepted_record_unchanged_against_manifest": True,
        "frozen_inputs_unchanged": inputs, "source_changes": source_changes,
        "ledger_rows": 58, "ledger_differences": ledger_diffs, "workbook_comparison": views,
        "pdf": {"pages": 1, "page_dimensions_unchanged": True, "text_only_change": "Extended Data Fig. 9l -> Extended Data Fig. 9o"},
        "scientific_intervals_cohorts_summary_statistics_unchanged": True,
        "visual_qa": {"status": "PASS", "reviewed": [
            "build/preview_table.png", "build/preview_source.png", "build/preview_summary.png",
            "../table2_audit/previews/before_publication.png", "../table2_audit/previews/before_short_note.png",
            "../table2_audit/previews/after_short_note.png", "../table2_audit/previews/after_pdf_page.png"],
            "result": "Updated references are legible; no clipping, overlap, or new layout defect."},
        "outputs": source_build["outputs"], "uploaded": False, "accepted_record_promoted": False,
    }
    target = audit / "IMPLEMENTATION_VALIDATION.json"
    target.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt": str(target), "sha256": sha256(target)}, indent=2))


if __name__ == "__main__":
    main()
