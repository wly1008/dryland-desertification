"""Estimate fixed indicator normalization bounds."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, UPSTREAM_DATA_ROOT
import math
import os
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
stats_root = str(parameter_file('01_upstream_reference/statistics'))
out_path = str(UPSTREAM_DATA_ROOT / 'statistics/index_fixed_bounds.xlsx')
indices = ['Albedo', 'DFI', 'LST', 'MSAVI', 'SMI', 'TGSI', 'Wet']
required_headers = ['Filename', 'Year', 'Month', 'P1', 'P99']
bounds = []
for index_name in indices:
    stats_path = str(parameter_file(f'01_upstream_reference/statistics/{index_name}_stats.xlsx'))
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f'Statistics workbook not found: {stats_path}')
    source_wb = load_workbook(stats_path, read_only=True, data_only=True)
    source_ws = source_wb.active
    header_values = [cell.value for cell in next(source_ws.iter_rows(min_row=1, max_row=1))]
    header_index = {value: position for position, value in enumerate(header_values)}
    legacy_layout = len(header_values) == 7 and header_values[5:7] == ['P1', 'P99']
    if legacy_layout and all(not str(value).isascii() for value in header_values[:3]):
        header_index.update({'Filename': 0, 'Year': 1, 'Month': 2})
    missing_headers = [name for name in required_headers if name not in header_index]
    if missing_headers:
        source_wb.close()
        raise ValueError(f"{stats_path} lacks required fields: {', '.join(missing_headers)}")
    min_p1 = math.inf
    max_p99 = -math.inf
    min_p1_file = None
    max_p99_file = None
    valid_count = 0
    year_months = []
    for row_number, row in enumerate(source_ws.iter_rows(min_row=2, values_only=True), start=2):
        file_name = row[header_index['Filename']]
        if file_name in (None, ''):
            continue
        p1_value = row[header_index['P1']]
        p99_value = row[header_index['P99']]
        try:
            p1_value = float(p1_value)
            p99_value = float(p99_value)
        except (TypeError, ValueError) as exc:
            source_wb.close()
            raise ValueError(f'{stats_path} row {row_number} row P1/P99 is not numeric') from exc
        if not (math.isfinite(p1_value) and math.isfinite(p99_value)):
            source_wb.close()
            raise ValueError(f'{stats_path} row {row_number} row P1/P99 contains NaN or infinity')
        year_value = row[header_index['Year']]
        month_value = row[header_index['Month']]
        try:
            year_months.append((int(year_value), int(month_value)))
        except (TypeError, ValueError) as exc:
            source_wb.close()
            raise ValueError(f'{stats_path} row {row_number} row year or month is invalid') from exc
        valid_count += 1
        if p1_value < min_p1:
            min_p1 = p1_value
            min_p1_file = str(file_name)
        if p99_value > max_p99:
            max_p99 = p99_value
            max_p99_file = str(file_name)
    source_wb.close()
    if valid_count == 0:
        raise ValueError(f'{stats_path} has no valid statistical records')
    if max_p99 <= min_p1:
        raise ValueError(f'{index_name} fixed upper bound ({max_p99}) does not exceed lower bound ({min_p1})')
    start_year, start_month = min(year_months)
    end_year, end_month = max(year_months)
    bounds.append([index_name, min_p1, max_p99, min_p1_file, max_p99_file, valid_count, f'{start_year:04d}-{start_month:02d}', f'{end_year:04d}-{end_month:02d}', os.path.basename(stats_path)])
os.makedirs(os.path.dirname(out_path), exist_ok=True)
tmp_path = out_path + '.tmp.xlsx'
out_wb = Workbook()
out_ws = out_wb.active
out_ws.title = 'fixed_bounds'
out_ws.append(['Indicator', 'Fixed lower bound (minimum P1)', 'Fixed upper bound (maximum P99)', 'Lower-bound source file', 'Upper-bound source file', 'Valid record count', 'First year-month', 'Last year-month', 'Source statistics workbook'])
for record in bounds:
    out_ws.append(record)
header_fill = PatternFill('solid', fgColor='1F4E78')
header_font = Font(color='FFFFFF', bold=True)
thin_gray = Side(style='thin', color='D9E2F3')
for cell in out_ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border = Border(bottom=thin_gray)
for row in out_ws.iter_rows(min_row=2, max_row=out_ws.max_row):
    for cell in row:
        cell.alignment = Alignment(vertical='center')
    for cell in (row[1], row[2]):
        if float(cell.value).is_integer():
            cell.number_format = '0'
        else:
            cell.number_format = '0.###############'
    row[5].number_format = '0'
column_widths = [12, 23, 23, 24, 24, 13, 13, 13, 22]
for column_number, width in enumerate(column_widths, start=1):
    out_ws.column_dimensions[get_column_letter(column_number)].width = width
out_ws.freeze_panes = 'A2'
out_ws.auto_filter.ref = out_ws.dimensions
out_ws.row_dimensions[1].height = 24
out_ws.sheet_view.showGridLines = False
note_ws = out_wb.create_sheet('Description')
note_rows = [['Item', 'Content'], ['Calculation rule', 'For each indicator, take the minimum P1 across all records as its fixed lower bound and the maximum P99 as its fixed upper bound.'], ['Source data directory', stats_root], ['Indicator count', len(indices)], ['Generated at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]]
for row in note_rows:
    note_ws.append(row)
for cell in note_ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center')
note_ws.column_dimensions['A'].width = 18
note_ws.column_dimensions['B'].width = 100
for row in note_ws.iter_rows():
    for cell in row:
        cell.alignment = Alignment(vertical='top', wrap_text=True)
note_ws.freeze_panes = 'A2'
note_ws.sheet_view.showGridLines = False
out_wb.save(tmp_path)
out_wb.close()
os.replace(tmp_path, out_path)
print(f'Written: {out_path}')
for record in bounds:
    print(f'{record[0]:<7}  fixed lower bound={record[1]:.15g}  fixed upper bound={record[2]:.15g}  record count={record[5]}')
