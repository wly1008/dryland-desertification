"""Summarize knowledge-granulation class frequencies."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
import numpy as np
import pandas as pd
import rasterio
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from tqdm import tqdm
import config as cfg
if not cfg.VALID_PIXEL_COUNT_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 4 valid-pixel summary workbook: {cfg.VALID_PIXEL_COUNT_XLSX}')
valid_pixel_table = pd.read_excel(cfg.VALID_PIXEL_COUNT_XLSX, sheet_name='valid_pixel_count', engine=cfg.EXCEL_ENGINE)
required_valid_pixel_columns = {'year', 'valid_pixels', 'indicator_count', 'indicators', 'statistic'}
missing_valid_pixel_columns = required_valid_pixel_columns - set(valid_pixel_table.columns)
if missing_valid_pixel_columns:
    raise RuntimeError(f'Step 4 summary lacks required fields: {sorted(missing_valid_pixel_columns)}')
if len(valid_pixel_table) != len(cfg.YEARS):
    raise RuntimeError(f'Step 4 row count differs from configured years; required {len(cfg.YEARS)}; actual: {len(valid_pixel_table)}')
if set(valid_pixel_table['year'].astype(int)) != set(cfg.YEARS):
    raise RuntimeError('Step 4 valid-pixel summary years do not match YEARS')
if valid_pixel_table['year'].astype(int).duplicated().any():
    raise RuntimeError('Step 4 valid-pixel summary contains duplicate years')
expected_indicator_text = ','.join(cfg.PREPROCESS_INDICATORS)
if not valid_pixel_table['indicators'].astype(str).eq(expected_indicator_text).all():
    raise RuntimeError('Step 4 indicator names or order do not match PREPROCESS_INDICATORS')
if not valid_pixel_table['indicator_count'].astype(int).eq(len(cfg.PREPROCESS_INDICATORS)).all():
    raise RuntimeError('Step 4 indicator count differs from the current configuration')
if not valid_pixel_table['statistic'].astype(str).eq(cfg.STATISTIC_NAME).all():
    raise RuntimeError('Step 4 statistic type differs from the current configuration')
expected_valid_count_by_year = {int(row['year']): int(row['valid_pixels']) for _, row in valid_pixel_table.iterrows()}
jobs = []
for year in cfg.YEARS:
    jobs.append({'year': year, 'mask_path': cfg.VALID_MASK_OUTPUT_DIR / cfg.MASK_FILE_PATTERN.format(year=year), 'class_paths': {indicator: cfg.GRANULATION_OUTPUT_DIR / indicator / cfg.GRANULATION_FILE_PATTERN.format(indicator=indicator, year=year) for indicator in cfg.PREPROCESS_INDICATORS}})
missing_inputs = []
for job in jobs:
    if not job['mask_path'].is_file():
        missing_inputs.append(job['mask_path'])
    for class_path in job['class_paths'].values():
        if not class_path.is_file():
            missing_inputs.append(class_path)
if missing_inputs:
    missing_preview = '\n'.join((str(path) for path in missing_inputs[:20]))
    raise FileNotFoundError(f'Step 5 is missing {len(missing_inputs)} input files:\n{missing_preview}')
formal_output_paths = [cfg.CLASS_COUNTS_XLSX, cfg.CLASS_PROBABILITIES_XLSX]
existing_outputs = [path for path in formal_output_paths if path.exists()]
if existing_outputs and (not cfg.OVERWRITE):
    existing_preview = '\n'.join((str(path) for path in existing_outputs))
    raise FileExistsError(f'Step 5 workbook already exists and OVERWRITE=False:\n{existing_preview}')
counts_by_year = {}
probabilities_by_year = {}
valid_count_by_year = {}
for job in tqdm(jobs, desc='Calculate annual class counts and probabilities', unit='year', dynamic_ncols=True):
    year = job['year']
    mask_path = job['mask_path']
    with rasterio.open(mask_path) as mask_source:
        mask_tags = mask_source.tags()
        mask_image_structure = mask_source.tags(1, ns='IMAGE_STRUCTURE')
        mask_ok = mask_source.count == 1 and mask_source.dtypes[0] == cfg.MASK_DTYPE and (mask_source.nodata == cfg.MASK_NODATA) and (mask_image_structure.get('NBITS') == str(cfg.MASK_NBITS)) and (mask_tags.get('PROCESS_STEP') == '04valid_mask') and (mask_tags.get('YEAR') == str(year)) and (mask_tags.get('STATISTIC') == cfg.STATISTIC_NAME) and (mask_tags.get('PREPROCESS_INDICATORS') == expected_indicator_text) and (mask_tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS))) and (mask_tags.get('VALIDITY_SOURCE') == 'knowledge_granulation_classes')
        if not mask_ok:
            raise RuntimeError(f'Annual common-valid mask failed structure or metadata checks: {mask_path}')
        valid_pixel_count = 0
        invalid_mask_code_count = 0
        for _, window in mask_source.block_windows(1):
            mask_data = mask_source.read(1, window=window, masked=False)
            valid_pixel_count += int(np.count_nonzero(mask_data == cfg.MASK_VALID_VALUE))
            invalid_mask_code_count += int(np.count_nonzero((mask_data != cfg.MASK_INVALID_VALUE) & (mask_data != cfg.MASK_VALID_VALUE)))
        if invalid_mask_code_count:
            raise RuntimeError(f'{year} annual mask contains {invalid_mask_code_count} codes other than 0 or 1')
        if valid_pixel_count <= 0:
            raise RuntimeError(f'{year} annual common-valid pixel count is zero; probabilities are undefined')
        if valid_pixel_count != expected_valid_count_by_year[year]:
            raise RuntimeError(f'{year} annual mask count differs from the step 4 summary; raster {valid_pixel_count}; table {expected_valid_count_by_year[year]}')
        valid_count_by_year[year] = valid_pixel_count
        annual_counts = {}
        annual_probabilities = {}
        indicator_progress = tqdm(cfg.PREPROCESS_INDICATORS, desc=f'{year} annual indicator ', unit='indicator', leave=False, dynamic_ncols=True)
        for indicator in indicator_progress:
            class_path = job['class_paths'][indicator]
            with rasterio.open(class_path) as class_source:
                class_tags = class_source.tags()
                transform_matches = np.allclose(np.asarray(tuple(class_source.transform), dtype=np.float64), np.asarray(tuple(mask_source.transform), dtype=np.float64), rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
                class_ok = class_source.count == 1 and class_source.width == mask_source.width and (class_source.height == mask_source.height) and (class_source.crs == mask_source.crs) and transform_matches and (class_source.dtypes[0] == cfg.RECLASS_DTYPE) and (class_source.nodata == cfg.RECLASS_NODATA) and (class_tags.get('PROCESS_STEP') == '03knowledge_granulation') and (class_tags.get('INDICATOR') == indicator) and (class_tags.get('YEAR') == str(year)) and (class_tags.get('BIN_COUNT') == str(cfg.BIN_COUNT)) and (class_tags.get('PREPROCESS_INDICATORS') == expected_indicator_text) and (class_tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS)))
                if not class_ok:
                    raise RuntimeError(f'Reclassified raster failed structure, grid, or metadata checks: {class_path}')
                class_counts = np.zeros(cfg.BIN_COUNT, dtype=np.int64)
                invalid_class_code_count = 0
                for _, window in mask_source.block_windows(1):
                    mask_data = mask_source.read(1, window=window, masked=False)
                    common_valid = mask_data == cfg.MASK_VALID_VALUE
                    if not bool(common_valid.any()):
                        continue
                    class_data = class_source.read(1, window=window, masked=False)
                    valid_area_classes = class_data[common_valid]
                    legal_codes = (valid_area_classes >= cfg.RECLASS_MIN_VALUE) & (valid_area_classes <= cfg.RECLASS_MAX_VALUE)
                    invalid_class_code_count += int(np.count_nonzero(~legal_codes))
                    if bool(legal_codes.any()):
                        block_counts = np.bincount(valid_area_classes[legal_codes].astype(np.int64), minlength=cfg.BIN_COUNT + 1)
                        class_counts += block_counts[cfg.RECLASS_MIN_VALUE:cfg.RECLASS_MAX_VALUE + 1]
                if invalid_class_code_count:
                    raise RuntimeError(f'{indicator} {year} contains within the common-valid mask {invalid_class_code_count} zero, NoData, or out-of-range class values')
                counted_pixel_count = int(class_counts.sum())
                if counted_pixel_count != valid_pixel_count:
                    raise RuntimeError(f'{indicator} {year} class sum differs from the common-valid pixel count; class sum {counted_pixel_count}; valid count {valid_pixel_count}')
                class_probabilities = class_counts.astype(np.float64) / valid_pixel_count
                probability_sum = float(class_probabilities.sum())
                if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=cfg.PROBABILITY_SUM_TOLERANCE):
                    raise RuntimeError(f'{indicator} {year} class probabilities do not sum to one: {probability_sum:.17g}')
                annual_counts[indicator] = class_counts
                annual_probabilities[indicator] = class_probabilities
        counts_by_year[year] = annual_counts
        probabilities_by_year[year] = annual_probabilities
count_tables = {}
probability_tables = {}
class_index = list(range(cfg.RECLASS_MIN_VALUE, cfg.RECLASS_MAX_VALUE + 1))
study_period_valid_count = sum(valid_count_by_year.values())
if study_period_valid_count <= 0:
    raise RuntimeError('Zero common-valid pixels in the study period; cannot calculate pooled probabilities')
study_period_counts = {}
study_period_probabilities = {}
for indicator in cfg.PREPROCESS_INDICATORS:
    indicator_count_stack = np.stack([counts_by_year[year][indicator] for year in cfg.YEARS], axis=0)
    total_class_counts = indicator_count_stack.sum(axis=0, dtype=np.int64)
    if int(total_class_counts.sum()) != study_period_valid_count:
        raise RuntimeError(f'{indicator} study-period class sum differs from total valid pixels')
    total_class_probabilities = total_class_counts.astype(np.float64) / study_period_valid_count
    if not np.isclose(float(total_class_probabilities.sum()), 1.0, rtol=0.0, atol=cfg.PROBABILITY_SUM_TOLERANCE):
        raise RuntimeError(f'{indicator} study-period class probabilities do not sum to one')
    study_period_counts[indicator] = total_class_counts
    study_period_probabilities[indicator] = total_class_probabilities
output_sheet_keys = [*cfg.YEARS, cfg.STUDY_PERIOD_SHEET_NAME]
counts_by_sheet = {**counts_by_year, cfg.STUDY_PERIOD_SHEET_NAME: study_period_counts}
probabilities_by_sheet = {**probabilities_by_year, cfg.STUDY_PERIOD_SHEET_NAME: study_period_probabilities}
valid_count_by_sheet = {**valid_count_by_year, cfg.STUDY_PERIOD_SHEET_NAME: study_period_valid_count}
for sheet_key in output_sheet_keys:
    count_table = pd.DataFrame({indicator: counts_by_sheet[sheet_key][indicator] for indicator in cfg.PREPROCESS_INDICATORS}, index=class_index)
    count_table.loc[cfg.EXCEL_COUNT_ROW_LABEL] = [valid_count_by_sheet[sheet_key] for _ in cfg.PREPROCESS_INDICATORS]
    count_table.index.name = cfg.EXCEL_INDEX_LABEL
    count_tables[sheet_key] = count_table
    probability_table = pd.DataFrame({indicator: probabilities_by_sheet[sheet_key][indicator] for indicator in cfg.PREPROCESS_INDICATORS}, index=class_index)
    probability_table.loc[cfg.EXCEL_PROBABILITY_SUM_ROW_LABEL] = [float(probability_table[indicator].sum()) for indicator in cfg.PREPROCESS_INDICATORS]
    probability_table.index.name = cfg.EXCEL_INDEX_LABEL
    probability_tables[sheet_key] = probability_table
cfg.CLASS_STATISTICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
count_temporary_path = cfg.CLASS_COUNTS_XLSX.with_name(f'{cfg.CLASS_COUNTS_XLSX.stem}.part{cfg.CLASS_COUNTS_XLSX.suffix}')
probability_temporary_path = cfg.CLASS_PROBABILITIES_XLSX.with_name(f'{cfg.CLASS_PROBABILITIES_XLSX.stem}.part{cfg.CLASS_PROBABILITIES_XLSX.suffix}')
count_temporary_path.unlink(missing_ok=True)
probability_temporary_path.unlink(missing_ok=True)
count_check = None
probability_check = None
header_fill = PatternFill(fill_type='solid', fgColor='1F4E78')
header_font = Font(color='FFFFFF', bold=True)
total_fill = PatternFill(fill_type='solid', fgColor='D9EAF7')
total_font = Font(bold=True)
separator_side = Side(style='thin', color='7F8C8D')
total_border = Border(top=separator_side)
try:
    with pd.ExcelWriter(count_temporary_path, engine=cfg.EXCEL_ENGINE) as count_writer:
        for sheet_key in output_sheet_keys:
            sheet_name = str(sheet_key)
            count_tables[sheet_key].to_excel(count_writer, sheet_name=sheet_name, index=True)
            worksheet = count_writer.sheets[sheet_name]
            worksheet.freeze_panes = 'B2'
            worksheet.sheet_view.showGridLines = False
            worksheet.auto_filter.ref = f'A1:{get_column_letter(len(cfg.PREPROCESS_INDICATORS) + 1)}{cfg.BIN_COUNT + 1}'
            worksheet.column_dimensions['A'].width = 12
            for column_index in range(2, len(cfg.PREPROCESS_INDICATORS) + 2):
                column_letter = worksheet.cell(1, column_index).column_letter
                worksheet.column_dimensions[column_letter].width = 14
                for row_index in range(2, cfg.BIN_COUNT + 3):
                    worksheet.cell(row_index, column_index).number_format = '#,##0'
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
            total_row_number = cfg.BIN_COUNT + 2
            for cell in worksheet[total_row_number]:
                cell.fill = total_fill
                cell.font = total_font
                cell.border = total_border
    with pd.ExcelWriter(probability_temporary_path, engine=cfg.EXCEL_ENGINE) as probability_writer:
        for sheet_key in output_sheet_keys:
            sheet_name = str(sheet_key)
            probability_tables[sheet_key].to_excel(probability_writer, sheet_name=sheet_name, index=True)
            worksheet = probability_writer.sheets[sheet_name]
            worksheet.freeze_panes = 'B2'
            worksheet.sheet_view.showGridLines = False
            worksheet.auto_filter.ref = f'A1:{get_column_letter(len(cfg.PREPROCESS_INDICATORS) + 1)}{cfg.BIN_COUNT + 1}'
            worksheet.column_dimensions['A'].width = 12
            for column_index in range(2, len(cfg.PREPROCESS_INDICATORS) + 2):
                column_letter = worksheet.cell(1, column_index).column_letter
                worksheet.column_dimensions[column_letter].width = 16
                for row_index in range(2, cfg.BIN_COUNT + 3):
                    worksheet.cell(row_index, column_index).number_format = '0.0000000000'
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
            total_row_number = cfg.BIN_COUNT + 2
            for cell in worksheet[total_row_number]:
                cell.fill = total_fill
                cell.font = total_font
                cell.border = total_border
    count_check = load_workbook(count_temporary_path, read_only=True, data_only=True)
    probability_check = load_workbook(probability_temporary_path, read_only=True, data_only=True)
    expected_sheet_names = [str(sheet_key) for sheet_key in output_sheet_keys]
    if count_check.sheetnames != expected_sheet_names:
        raise RuntimeError('Incorrect annual sheet names or order in the count workbook')
    if probability_check.sheetnames != expected_sheet_names:
        raise RuntimeError('Incorrect annual sheet names or order in the probability workbook')
    expected_headers = [cfg.EXCEL_INDEX_LABEL, *cfg.PREPROCESS_INDICATORS]
    expected_count_labels = [*class_index, cfg.EXCEL_COUNT_ROW_LABEL]
    expected_probability_labels = [*class_index, cfg.EXCEL_PROBABILITY_SUM_ROW_LABEL]
    for sheet_key in output_sheet_keys:
        count_sheet = count_check[str(sheet_key)]
        probability_sheet = probability_check[str(sheet_key)]
        count_headers = [count_sheet.cell(1, column).value for column in range(1, len(expected_headers) + 1)]
        probability_headers = [probability_sheet.cell(1, column).value for column in range(1, len(expected_headers) + 1)]
        if count_headers != expected_headers:
            raise RuntimeError(f'{sheet_key} count-table headers are incorrect')
        if probability_headers != expected_headers:
            raise RuntimeError(f'{sheet_key} probability-table headers are incorrect')
        count_labels = [count_sheet.cell(row, 1).value for row in range(2, cfg.BIN_COUNT + 3)]
        probability_labels = [probability_sheet.cell(row, 1).value for row in range(2, cfg.BIN_COUNT + 3)]
        if count_labels != expected_count_labels:
            raise RuntimeError(f'{sheet_key} count-table index column is incorrect')
        if probability_labels != expected_probability_labels:
            raise RuntimeError(f'{sheet_key} probability-table index column is incorrect')
        for indicator_offset, indicator in enumerate(cfg.PREPROCESS_INDICATORS, start=2):
            excel_counts = np.asarray([count_sheet.cell(row, indicator_offset).value for row in range(2, cfg.BIN_COUNT + 2)], dtype=np.int64)
            excel_count_total = int(count_sheet.cell(cfg.BIN_COUNT + 2, indicator_offset).value)
            if not np.array_equal(excel_counts, counts_by_sheet[sheet_key][indicator]):
                raise RuntimeError(f'{indicator} {sheet_key} count-table written values differ from in-memory counts')
            if excel_count_total != valid_count_by_sheet[sheet_key]:
                raise RuntimeError(f'{indicator} {sheet_key} count-table count row is incorrect')
            excel_probabilities = np.asarray([probability_sheet.cell(row, indicator_offset).value for row in range(2, cfg.BIN_COUNT + 2)], dtype=np.float64)
            excel_probability_sum = float(probability_sheet.cell(cfg.BIN_COUNT + 2, indicator_offset).value)
            if not np.allclose(excel_probabilities, probabilities_by_sheet[sheet_key][indicator], rtol=0.0, atol=np.finfo(np.float64).eps):
                raise RuntimeError(f'{indicator} {sheet_key} probability-table written values differ from in-memory probabilities')
            if not np.isclose(excel_probability_sum, 1.0, rtol=0.0, atol=cfg.PROBABILITY_SUM_TOLERANCE):
                raise RuntimeError(f'{indicator} {sheet_key} probability-table sum row does not equal one')
    count_check.close()
    probability_check.close()
    os.replace(count_temporary_path, cfg.CLASS_COUNTS_XLSX)
    os.replace(probability_temporary_path, cfg.CLASS_PROBABILITIES_XLSX)
except Exception:
    if count_check is not None:
        count_check.close()
    if probability_check is not None:
        probability_check.close()
    count_temporary_path.unlink(missing_ok=True)
    probability_temporary_path.unlink(missing_ok=True)
    raise
print(f'Class-count and probability summaries completed: {len(cfg.YEARS)} years; {len(cfg.PREPROCESS_INDICATORS)} indicators; {cfg.BIN_COUNT} classes.')
print(f'Study-period summary sheet: {cfg.STUDY_PERIOD_SHEET_NAME}')
print(f'Class-count workbook: {cfg.CLASS_COUNTS_XLSX}')
print(f'Class-probability workbook: {cfg.CLASS_PROBABILITIES_XLSX}')
