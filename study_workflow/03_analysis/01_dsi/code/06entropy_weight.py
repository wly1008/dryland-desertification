"""Calculate indicator entropy weights from class statistics."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from tqdm import tqdm
import config as cfg
if not cfg.CLASS_COUNTS_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 5 annual class-count workbook: {cfg.CLASS_COUNTS_XLSX}')
if cfg.GK_ER_W_XLSX.exists() and (not cfg.OVERWRITE):
    raise FileExistsError(f'Step 6 workbook already exists and OVERWRITE=False: {cfg.GK_ER_W_XLSX}')
calculation_indicators = cfg.SELECTED_INDICATORS_IN_CONFIG_ORDER
calculation_sheet_keys = [*cfg.YEARS, cfg.STUDY_PERIOD_SHEET_NAME]
if len(calculation_indicators) < 2:
    raise ValueError('Entropy weighting requires at least two selected indicators')
if cfg.BIN_COUNT < 2:
    raise ValueError('Knowledge-granulation BIN_COUNT must be at least two')
count_workbook = pd.ExcelFile(cfg.CLASS_COUNTS_XLSX, engine=cfg.EXCEL_ENGINE)
expected_sheet_names = [str(sheet_key) for sheet_key in calculation_sheet_keys]
if count_workbook.sheet_names != expected_sheet_names:
    raise RuntimeError(f'class_counts.xlsx sheet names or order differ from configuration.\nRequired: {expected_sheet_names}\nActual: {count_workbook.sheet_names}')
gk_table = pd.DataFrame(index=list(calculation_indicators), columns=list(calculation_sheet_keys), dtype=np.float64)
er_table = pd.DataFrame(index=list(calculation_indicators), columns=list(calculation_sheet_keys), dtype=np.float64)
weight_table = pd.DataFrame(index=list(calculation_indicators), columns=list(calculation_sheet_keys), dtype=np.float64)
valid_count_by_sheet = {}
class_counts_by_sheet = {}
expected_index_labels = [*[str(level) for level in range(cfg.RECLASS_MIN_VALUE, cfg.RECLASS_MAX_VALUE + 1)], cfg.EXCEL_COUNT_ROW_LABEL]
for sheet_key in tqdm(calculation_sheet_keys, desc='Calculate Gk, Er, and W', unit='sheet', dynamic_ncols=True):
    period_count_table = pd.read_excel(count_workbook, sheet_name=str(sheet_key), index_col=0)
    if str(period_count_table.index.name) != cfg.EXCEL_INDEX_LABEL:
        raise RuntimeError(f'{sheet_key} count-table first-column title must be {cfg.EXCEL_INDEX_LABEL}：{period_count_table.index.name}')
    period_count_table.index.name = cfg.EXCEL_INDEX_LABEL
    actual_columns = [str(column) for column in period_count_table.columns]
    expected_columns = list(cfg.PREPROCESS_INDICATORS)
    if actual_columns != expected_columns:
        raise RuntimeError(f'{sheet_key} count-table indicators or order differ from PREPROCESS_INDICATORS.\nRequired: {expected_columns}\nActual: {actual_columns}')
    selected_count_table = period_count_table.loc[:, list(calculation_indicators)]
    normalized_index_labels = []
    for label in selected_count_table.index:
        if isinstance(label, (int, np.integer)):
            normalized_index_labels.append(str(int(label)))
        elif isinstance(label, (float, np.floating)) and float(label).is_integer():
            normalized_index_labels.append(str(int(label)))
        else:
            normalized_index_labels.append(str(label).strip())
    if normalized_index_labels != expected_index_labels:
        raise RuntimeError(f'{sheet_key} count-table index rows are incorrect.\nRequired: {expected_index_labels}\nActual: {normalized_index_labels}')
    raw_class_counts = selected_count_table.iloc[:cfg.BIN_COUNT].to_numpy(dtype=np.float64)
    raw_total_counts = selected_count_table.iloc[cfg.BIN_COUNT].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(raw_class_counts).all()):
        raise RuntimeError(f'{sheet_key} class counts contain NaN or Inf')
    if not bool(np.isfinite(raw_total_counts).all()):
        raise RuntimeError(f'{sheet_key} count row contains NaN or Inf')
    if bool((raw_class_counts < 0).any()):
        raise RuntimeError(f'{sheet_key} class counts contain negative values')
    if bool((raw_total_counts <= 1).any()):
        raise RuntimeError(f'{sheet_key} count must be an integer greater than one')
    if not bool(np.equal(raw_class_counts, np.floor(raw_class_counts)).all()):
        raise RuntimeError(f'{sheet_key} class counts contain non-integer values')
    if not bool(np.equal(raw_total_counts, np.floor(raw_total_counts)).all()):
        raise RuntimeError(f'{sheet_key} count row contains non-integer values')
    class_counts = raw_class_counts.astype(np.int64)
    total_counts = raw_total_counts.astype(np.int64)
    period_valid_count = int(total_counts[0])
    if not bool(np.equal(total_counts, period_valid_count).all()):
        raise RuntimeError(f'{sheet_key} indicator counts differ: {total_counts.tolist()}')
    class_count_sums = class_counts.sum(axis=0, dtype=np.int64)
    if not bool(np.equal(class_count_sums, period_valid_count).all()):
        raise RuntimeError(f'{sheet_key} some class sums differ from count.\nClass sums: {class_count_sums.tolist()}\ncount：{period_valid_count}')
    valid_count_by_sheet[sheet_key] = period_valid_count
    class_counts_by_sheet[sheet_key] = class_counts
    probabilities = class_counts.astype(np.float64) / period_valid_count
    probability_sums = probabilities.sum(axis=0, dtype=np.float64)
    if not bool(np.isclose(probability_sums, 1.0, rtol=0.0, atol=cfg.PROBABILITY_SUM_TOLERANCE).all()):
        raise RuntimeError(f'{sheet_key} probabilities reconstructed from counts do not sum to one: {probability_sums.tolist()}')
    period_gk = np.square(probabilities).sum(axis=0, dtype=np.float64)
    entropy_terms = np.zeros_like(probabilities, dtype=np.float64)
    positive_count_mask = class_counts > 0
    entropy_terms[positive_count_mask] = probabilities[positive_count_mask] * np.log2(class_counts[positive_count_mask].astype(np.float64))
    period_er = entropy_terms.sum(axis=0, dtype=np.float64) / np.log2(float(period_valid_count))
    if not bool(np.isfinite(period_gk).all()):
        raise RuntimeError(f'{sheet_key} Gk contains NaN or Inf')
    if not bool(np.isfinite(period_er).all()):
        raise RuntimeError(f'{sheet_key} Er contains NaN or Inf')
    gk_in_range = (period_gk >= -cfg.VALUE_RANGE_TOLERANCE) & (period_gk <= 1.0 + cfg.VALUE_RANGE_TOLERANCE)
    er_in_range = (period_er >= -cfg.VALUE_RANGE_TOLERANCE) & (period_er <= 1.0 + cfg.VALUE_RANGE_TOLERANCE)
    if not bool(gk_in_range.all()):
        raise RuntimeError(f'{sheet_key} Gk is outside [0, 1]: {period_gk.tolist()}')
    if not bool(er_in_range.all()):
        raise RuntimeError(f'{sheet_key} Er is outside [0, 1]: {period_er.tolist()}')
    period_gk = np.clip(period_gk, 0.0, 1.0)
    period_er = np.clip(period_er, 0.0, 1.0)
    period_redundancy = 1.0 - period_er
    weight_denominator = float(period_redundancy.sum(dtype=np.float64))
    equivalent_denominator = float(len(calculation_indicators) - period_er.sum(dtype=np.float64))
    if not np.isclose(weight_denominator, equivalent_denominator, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE):
        raise RuntimeError(f'{sheet_key} the two weight-denominator calculations disagree')
    if weight_denominator <= cfg.WEIGHT_SUM_TOLERANCE:
        raise RuntimeError(f'{sheet_key} weight denominator is zero or too small: {weight_denominator:.17g}')
    period_weights = period_redundancy / weight_denominator
    if not bool(np.isfinite(period_weights).all()):
        raise RuntimeError(f'{sheet_key} weights contain NaN or Inf')
    if bool(((period_weights < -cfg.VALUE_RANGE_TOLERANCE) | (period_weights > 1.0 + cfg.VALUE_RANGE_TOLERANCE)).any()):
        raise RuntimeError(f'{sheet_key} weights are outside [0, 1]: {period_weights.tolist()}')
    period_weight_sum = float(period_weights.sum(dtype=np.float64))
    if not np.isclose(period_weight_sum, 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE):
        raise RuntimeError(f'{sheet_key} indicator weights do not sum to one: {period_weight_sum:.17g}')
    gk_table.loc[:, sheet_key] = period_gk
    er_table.loc[:, sheet_key] = period_er
    weight_table.loc[:, sheet_key] = period_weights
expected_study_period_valid_count = sum((valid_count_by_sheet[year] for year in cfg.YEARS))
actual_study_period_valid_count = valid_count_by_sheet[cfg.STUDY_PERIOD_SHEET_NAME]
if actual_study_period_valid_count != expected_study_period_valid_count:
    raise RuntimeError(f'study_period count differs from the sum of annual counts; pooled {actual_study_period_valid_count}; annual total {expected_study_period_valid_count}')
expected_study_period_class_counts = np.stack([class_counts_by_sheet[year] for year in cfg.YEARS], axis=0).sum(axis=0, dtype=np.int64)
if not np.array_equal(class_counts_by_sheet[cfg.STUDY_PERIOD_SHEET_NAME], expected_study_period_class_counts):
    raise RuntimeError('study_period class counts do not equal the sum of annual class counts')
count_workbook.close()
for table_name, result_table in (('Gk', gk_table), ('Er', er_table), ('W', weight_table)):
    result_values = result_table.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(result_values).all()):
        raise RuntimeError(f'{table_name} summary table contains NaN or Inf')
    if bool(((result_values < -cfg.VALUE_RANGE_TOLERANCE) | (result_values > 1.0 + cfg.VALUE_RANGE_TOLERANCE)).any()):
        raise RuntimeError(f'{table_name} summary table contains values outside [0, 1]')
final_weight_sums = weight_table.sum(axis=0).to_numpy(dtype=np.float64)
if not bool(np.isclose(final_weight_sums, 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE).all()):
    raise RuntimeError(f'Worksheet W contains period-weight sums other than one: {final_weight_sums.tolist()}')
for result_table in (gk_table, er_table, weight_table):
    result_table.index.name = cfg.EXCEL_INDEX_LABEL
cfg.ENTROPY_WEIGHT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
temporary_output_path = cfg.GK_ER_W_XLSX.with_name(f'{cfg.GK_ER_W_XLSX.stem}.part{cfg.GK_ER_W_XLSX.suffix}')
temporary_output_path.unlink(missing_ok=True)
workbook_check = None
header_fill = PatternFill(fill_type='solid', fgColor='1F4E78')
header_font = Font(color='FFFFFF', bold=True)
index_fill = PatternFill(fill_type='solid', fgColor='D9EAF7')
index_font = Font(bold=True)
separator_side = Side(style='thin', color='7F8C8D')
index_border = Border(right=separator_side)
try:
    with pd.ExcelWriter(temporary_output_path, engine=cfg.EXCEL_ENGINE) as writer:
        for sheet_name, result_table in (('Gk', gk_table), ('Er', er_table), ('W', weight_table)):
            result_table.to_excel(writer, sheet_name=sheet_name, index=True)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = 'B2'
            worksheet.sheet_view.showGridLines = False
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.column_dimensions['A'].width = 14
            for column_index in range(2, len(calculation_sheet_keys) + 2):
                column_letter = get_column_letter(column_index)
                worksheet.column_dimensions[column_letter].width = 13
                for row_index in range(2, len(calculation_indicators) + 2):
                    worksheet.cell(row_index, column_index).number_format = '0.000000000000'
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
            for row_index in range(2, len(calculation_indicators) + 2):
                index_cell = worksheet.cell(row_index, 1)
                index_cell.fill = index_fill
                index_cell.font = index_font
                index_cell.border = index_border
    workbook_check = load_workbook(temporary_output_path, read_only=True, data_only=True)
    if workbook_check.sheetnames != ['Gk', 'Er', 'W']:
        raise RuntimeError(f'Step 6 temporary workbook has incorrect sheet names or order: {workbook_check.sheetnames}')
    expected_headers = [cfg.EXCEL_INDEX_LABEL, *calculation_sheet_keys]
    expected_row_labels = list(calculation_indicators)
    for sheet_name, expected_table in (('Gk', gk_table), ('Er', er_table), ('W', weight_table)):
        worksheet = workbook_check[sheet_name]
        actual_headers = [worksheet.cell(1, column).value for column in range(1, len(expected_headers) + 1)]
        actual_row_labels = [worksheet.cell(row, 1).value for row in range(2, len(expected_row_labels) + 2)]
        if actual_headers != expected_headers:
            raise RuntimeError(f'{sheet_name} worksheet headers are incorrect')
        if actual_row_labels != expected_row_labels:
            raise RuntimeError(f'{sheet_name} worksheet indicator rows or order are incorrect')
        excel_values = np.asarray([[worksheet.cell(row, column).value for column in range(2, len(calculation_sheet_keys) + 2)] for row in range(2, len(calculation_indicators) + 2)], dtype=np.float64)
        expected_values = expected_table.to_numpy(dtype=np.float64)
        if not bool(np.isfinite(excel_values).all()):
            raise RuntimeError(f'{sheet_name} written worksheet contains NaN or Inf')
        if not np.allclose(excel_values, expected_values, rtol=0.0, atol=np.finfo(np.float64).eps):
            raise RuntimeError(f'{sheet_name} worksheet values differ from in-memory calculations')
        if sheet_name == 'W':
            excel_weight_sums = excel_values.sum(axis=0, dtype=np.float64)
            if not bool(np.isclose(excel_weight_sums, 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE).all()):
                raise RuntimeError('Written W period-weight sums do not equal one')
    workbook_check.close()
    workbook_check = None
    os.replace(temporary_output_path, cfg.GK_ER_W_XLSX)
except Exception:
    if workbook_check is not None:
        workbook_check.close()
    temporary_output_path.unlink(missing_ok=True)
    raise
print(f'Input workbook: {cfg.CLASS_COUNTS_XLSX}')
print(f'Gk, Er, and W completed: {len(cfg.YEARS)} years, plus {cfg.STUDY_PERIOD_SHEET_NAME} summary column; {len(calculation_indicators)} indicators; {cfg.BIN_COUNT} classes.')
print(f'Current indicator group: {cfg.INDICATOR_GROUP_NAME}')
print(f'Output workbook: {cfg.GK_ER_W_XLSX}')
