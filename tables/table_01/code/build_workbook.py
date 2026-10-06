"""Build Supplementary Table 1 using openpyxl, without a private runtime."""
import argparse
import json
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


def build(output, source_data):
    source = json.loads((source_data / 'current_rows.json').read_text(encoding='utf-8'))
    book = Workbook()
    book._fonts[0] = Font(name='Carlito', size=11)  # Match the original workbook's default font.
    sheet = book.active
    sheet.title = 'Table S1'
    sheet.sheet_view.showGridLines = False
    for column, width in [('A', 42.5), ('B', 29.375), ('C', 45), ('D', 62.5)]:
        sheet.column_dimensions[column].width = width
    sheet.merge_cells('A1:D1')
    sheet['A1'] = source['title']
    sheet['A1'].font = Font(name='Arial', size=14, bold=True, color='FF111111')
    sheet.row_dimensions[1].height = 25.5
    sheet.row_dimensions[2].height = 7.5
    thin = Side(style='thin', color='FFC8C8C8')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for column, value in enumerate(source['headers'], 1):
        cell = sheet.cell(3, column, value)
        cell.font = Font(name='Arial', size=8, bold=True, color='FF222222')
        cell.fill = PatternFill('solid', fgColor='FFE4E4E4')
        cell.border = border
    sheet.row_dimensions[3].height = 21
    for index, row in enumerate(source['rows'], 4):
        values = [row['analysis'], row['panels'], f"n = {len(row['mice'])}\n" + ', '.join(row['mice']), row['note']]
        for column, value in enumerate(values, 1):
            cell = sheet.cell(index, column, value)
            cell.font = Font(name='Arial', size=8, color='FF222222', bold=False, italic=False)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            cell.border = border
    merges = []
    first = 0
    while first < len(source['rows']):
        last = first
        while last + 1 < len(source['rows']) and source['rows'][last + 1]['mice'] == source['rows'][first]['mice']:
            last += 1
        if last > first:
            address = f'C{first + 4}:C{last + 4}'
            sheet.merge_cells(address)
            sheet.cell(first + 4, 3).alignment = Alignment(wrap_text=True, vertical='center')
            merges.append(address)
        first = last + 1
    def wraps(value, width):
        return sum(math.ceil(len(line) / width) for line in str(value).split('\n'))
    for index, row in enumerate(source['rows']):
        shared = any(f'C{index + 4}' in merged for merged in sheet.merged_cells.ranges)
        lines = max(wraps(row['analysis'], 38), wraps(row['panels'], 25), wraps(row['note'], 59),
                    0 if shared else wraps(', '.join(row['mice']), 40) + 1)
        sheet.row_dimensions[index + 4].height = max(31, lines * 10.5 + 8)
    footnote = source.get('footnote', '')
    if footnote:
        footnote_row = len(source['rows']) + 5
        sheet.merge_cells(f'A{footnote_row}:D{footnote_row}')
        cell = sheet.cell(footnote_row, 1, footnote)
        cell.font = Font(name='Arial', size=8, italic=True, color='FF222222')
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        sheet.row_dimensions[footnote_row].height = max(21, wraps(footnote, 190) * 10.5 + 8)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'Supplementary_Table_S1_Mouse_Cohorts_by_Analysis.xlsx'
    book.save(path)
    (output / 'WORKBOOK_BUILD.json').write_text(json.dumps({
        'status': 'PASS', 'builder': 'openpyxl', 'rows': len(source['rows']), 'merges': merges,
        'formula_errors': 0, 'preview': 'Use the unchanged source-backed publication PDF builder.'
    }, indent=2) + '\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path, nargs='?', default=Path(__file__).resolve().parents[1] / 'build')
    parser.add_argument('--source-data', type=Path, default=Path(__file__).resolve().parents[1] / 'source_data')
    args = parser.parse_args()
    print(build(args.output_dir.resolve(), args.source_data.resolve()))
