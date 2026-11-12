"""Check DSI structure, masks, ranges, weights, and output values."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
from contextlib import ExitStack
import numpy as np
import pandas as pd
import rasterio
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from tqdm import tqdm
import config as cfg
if tuple(cfg.DI_OUTPUT_DIRS) != cfg.DI_WEIGHT_MODES:
    raise RuntimeError('DI_OUTPUT_DIRS does not match DI_WEIGHT_MODES')
if tuple(cfg.DI_QUALITY_OUTPUT_DIRS) != cfg.DI_WEIGHT_MODES:
    raise RuntimeError('DI_QUALITY_OUTPUT_DIRS does not match DI_WEIGHT_MODES')
if tuple(cfg.DI_QUALITY_XLSX_BY_MODE) != cfg.DI_WEIGHT_MODES:
    raise RuntimeError('DI_QUALITY_XLSX_BY_MODE does not match DI_WEIGHT_MODES')
if not cfg.GK_ER_W_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 6 Gk/Er/W workbook: {cfg.GK_ER_W_XLSX}')
existing_quality_workbooks = [cfg.DI_QUALITY_XLSX_BY_MODE[weight_mode] for weight_mode in cfg.DI_WEIGHT_MODES if cfg.DI_QUALITY_XLSX_BY_MODE[weight_mode].exists()]
if existing_quality_workbooks and (not cfg.OVERWRITE):
    existing_quality_text = '\n'.join((str(path) for path in existing_quality_workbooks))
    raise FileExistsError(f'Step 8 quality workbook already exists and OVERWRITE=False:\n{existing_quality_text}')
if tuple(cfg.DI_QUALITY_PERCENTILES) != (1.0, 5.0, 50.0, 95.0, 99.0):
    raise ValueError('Current summary fields require DI_QUALITY_PERCENTILES=(1.0, 5.0, 50.0, 95.0, 99.0)')
weight_table = pd.read_excel(cfg.GK_ER_W_XLSX, sheet_name='W', index_col=0, engine=cfg.EXCEL_ENGINE)
if str(weight_table.index.name) != cfg.EXCEL_INDEX_LABEL:
    raise RuntimeError('Worksheet W first-column title is incorrect')
calculation_indicators = cfg.SELECTED_INDICATORS_IN_CONFIG_ORDER
if [str(value) for value in weight_table.index] != list(calculation_indicators):
    raise RuntimeError('Worksheet W indicators or order differ from the current indicator group')
normalized_weight_columns = []
for value in weight_table.columns:
    if isinstance(value, (int, np.integer)):
        normalized_weight_columns.append(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        normalized_weight_columns.append(int(value))
    else:
        normalized_column = str(value).strip()
        if normalized_column == cfg.STUDY_PERIOD_SHEET_NAME:
            normalized_weight_columns.append(normalized_column)
        else:
            try:
                normalized_weight_columns.append(int(normalized_column))
            except ValueError as error:
                raise RuntimeError(f'Worksheet W has invalid period columns: {value}') from error
expected_weight_columns = [*cfg.YEARS, cfg.STUDY_PERIOD_SHEET_NAME]
if normalized_weight_columns != expected_weight_columns:
    raise RuntimeError('Worksheet W annual and study-period columns differ from configuration')
weight_table.columns = normalized_weight_columns
weight_values = weight_table.to_numpy(dtype=np.float64)
if not bool(np.isfinite(weight_values).all()):
    raise RuntimeError('Worksheet W contains NaN or Inf')
if not bool(np.isclose(weight_values.sum(axis=0, dtype=np.float64), 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE).all()):
    raise RuntimeError('Worksheet W period-weight sums do not equal one')
expected_file_names = {cfg.DI_FILE_PATTERN.format(year=year) for year in cfg.YEARS}
error_columns = ['year', 'severity', 'check', 'message', 'file']
error_rows_by_mode = {weight_mode: [] for weight_mode in cfg.DI_WEIGHT_MODES}
for weight_mode in cfg.DI_WEIGHT_MODES:
    di_output_dir = cfg.DI_OUTPUT_DIRS[weight_mode]
    actual_tif_paths = sorted(di_output_dir.glob('*.tif')) if di_output_dir.is_dir() else []
    part_file_names = sorted((path.name for path in di_output_dir.glob('*.part.tif'))) if di_output_dir.is_dir() else []
    actual_file_names = {path.name for path in actual_tif_paths if path.name not in part_file_names}
    extra_file_names = sorted(actual_file_names - expected_file_names)
    error_rows = error_rows_by_mode[weight_mode]
    for file_name in extra_file_names:
        error_rows.append({'year': None, 'severity': 'WARNING', 'check': 'unexpected_file', 'message': 'Unexpected GeoTIFF in the DSI output directory', 'file': str(di_output_dir / file_name)})
    for file_name in part_file_names:
        error_rows.append({'year': None, 'severity': 'ERROR', 'check': 'temporary_file', 'message': 'Residual .part.tif in output directory; a task may be incomplete', 'file': str(di_output_dir / file_name)})
summary_rows_by_mode = {weight_mode: [] for weight_mode in cfg.DI_WEIGHT_MODES}
quality_jobs = [(weight_mode, year) for weight_mode in cfg.DI_WEIGHT_MODES for year in cfg.YEARS]
for weight_mode, year in tqdm(quality_jobs, desc='Check both annual DSI modes', unit='file', dynamic_ncols=True):
    di_path = cfg.DI_OUTPUT_DIRS[weight_mode] / cfg.DI_FILE_PATTERN.format(year=year)
    mask_path = cfg.VALID_MASK_OUTPUT_DIR / cfg.MASK_FILE_PATTERN.format(year=year)
    if weight_mode == 'annual_weights':
        weight_column = year
    elif weight_mode == 'study_period_weights':
        weight_column = cfg.STUDY_PERIOD_SHEET_NAME
    else:
        raise ValueError(f'Unsupported DSI weighting mode: {weight_mode}')
    applied_weights = weight_table.loc[list(calculation_indicators), weight_column].to_numpy(dtype=np.float64)
    error_rows = error_rows_by_mode[weight_mode]
    summary_rows = summary_rows_by_mode[weight_mode]
    year_error_start = len(error_rows)
    summary_row = {'weight_mode': weight_mode, 'weight_column': str(weight_column), 'year': year, 'file': str(di_path), 'exists': di_path.is_file(), 'open_status': 'NOT_OPENED', 'status': 'ERROR', 'error_count': 0, 'warning_count': 0, 'total_pixels': np.nan, 'valid_pixels': np.nan, 'nodata_pixels': np.nan, 'valid_ratio': np.nan, 'mask_valid_pixels': np.nan, 'mask_mismatch_pixels': np.nan, 'zero_pixels': np.nan, 'nan_pixels': np.nan, 'inf_pixels': np.nan, 'below_0_pixels': np.nan, 'above_1_pixels': np.nan, 'minimum': np.nan, 'p01': np.nan, 'p05': np.nan, 'median': np.nan, 'p95': np.nan, 'p99': np.nan, 'maximum': np.nan, 'mean': np.nan, 'std': np.nan, 'driver': None, 'band_count': np.nan, 'dtype': None, 'nodata': np.nan, 'crs': None, 'width': np.nan, 'height': np.nan, 'x_resolution': np.nan, 'y_resolution': np.nan, 'transform': None, 'tiled': None, 'block_x_size': np.nan, 'block_y_size': np.nan, 'compression': None, 'predictor': None, 'process_step': None, 'statistic': None, 'period_kind': None, 'period_start': None, 'period_end': None, 'indicator_group': None, 'selected_indicator_count': np.nan, 'selected_indicators': None, 'preprocess_indicator_count': np.nan, 'preprocess_indicators': None, 'weight_sheet': None, 'metadata_weight_mode': None, 'metadata_weight_column': None, 'weight_sum': np.nan, 'weights_match': False, 'clipped_low_pixels': np.nan, 'clipped_high_pixels': np.nan, 'metadata_count_match': False, 'metadata_stats_match': False}
    if not di_path.is_file():
        error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'file_exists', 'message': 'Annual DSI file is missing', 'file': str(di_path)})
        summary_row['open_status'] = 'MISSING'
        summary_row['error_count'] = 1
        summary_rows.append(summary_row)
        continue
    try:
        with ExitStack() as stack:
            di_source = stack.enter_context(rasterio.open(di_path))
            mask_source = None
            if mask_path.is_file():
                mask_source = stack.enter_context(rasterio.open(mask_path))
            else:
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'mask_exists', 'message': 'Missing annual common-valid mask', 'file': str(mask_path)})
            summary_row['open_status'] = 'OK'
            tags = di_source.tags()
            image_structure = di_source.tags(ns='IMAGE_STRUCTURE')
            compression = di_source.compression.value.lower() if di_source.compression is not None else None
            block_y_size, block_x_size = di_source.block_shapes[0]
            summary_row.update({'total_pixels': di_source.width * di_source.height, 'driver': di_source.driver, 'band_count': di_source.count, 'dtype': di_source.dtypes[0], 'nodata': di_source.nodata, 'crs': str(di_source.crs), 'width': di_source.width, 'height': di_source.height, 'x_resolution': abs(float(di_source.transform.a)), 'y_resolution': abs(float(di_source.transform.e)), 'transform': ','.join((f'{value:.17g}' for value in tuple(di_source.transform))), 'tiled': bool(di_source.profile.get('tiled', False)), 'block_x_size': block_x_size, 'block_y_size': block_y_size, 'compression': compression, 'predictor': image_structure.get('PREDICTOR'), 'process_step': tags.get('PROCESS_STEP'), 'statistic': tags.get('STATISTIC'), 'period_kind': tags.get('PERIOD_KIND'), 'period_start': tags.get('PERIOD_START'), 'period_end': tags.get('PERIOD_END'), 'indicator_group': tags.get('INDICATOR_GROUP_NAME'), 'selected_indicator_count': tags.get('SELECTED_INDICATOR_COUNT'), 'selected_indicators': tags.get('SELECTED_INDICATORS'), 'preprocess_indicator_count': tags.get('PREPROCESS_INDICATOR_COUNT'), 'preprocess_indicators': tags.get('PREPROCESS_INDICATORS'), 'weight_sheet': tags.get('WEIGHT_SHEET'), 'metadata_weight_mode': tags.get('WEIGHT_MODE'), 'metadata_weight_column': tags.get('WEIGHT_COLUMN')})
            structural_checks = [('driver', di_source.driver == 'GTiff', 'Driver is not GTiff'), ('band_count', di_source.count == 1, 'Band count is not one'), ('dtype', di_source.dtypes[0] == cfg.DI_DTYPE, 'Data type is not Float32'), ('nodata', di_source.nodata is not None and np.isclose(di_source.nodata, cfg.DI_NODATA, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE), 'NoData is not 32767'), ('crs', str(di_source.crs) == cfg.EXPECTED_CRS, 'CRS is not EPSG:4326'), ('width', di_source.width == cfg.EXPECTED_WIDTH, 'Raster width is incorrect'), ('height', di_source.height == cfg.EXPECTED_HEIGHT, 'Raster height is incorrect'), ('x_resolution', np.isclose(abs(float(di_source.transform.a)), cfg.EXPECTED_X_RESOLUTION, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE), 'Longitude resolution is incorrect'), ('y_resolution', np.isclose(abs(float(di_source.transform.e)), cfg.EXPECTED_Y_RESOLUTION, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE), 'Latitude resolution is incorrect'), ('tiled', bool(di_source.profile.get('tiled', False)), 'GeoTIFF is not tiled'), ('block_size', block_x_size == cfg.GTIFF_BLOCK_X_SIZE and block_y_size == cfg.GTIFF_BLOCK_Y_SIZE, 'GeoTIFF tile size is not 256 by 256'), ('compression', compression == cfg.GTIFF_COMPRESSION.lower(), 'GeoTIFF compression is not DEFLATE'), ('predictor', image_structure.get('PREDICTOR') == str(cfg.FLOAT_PREDICTOR), 'Floating-point predictor is not three')]
            for check_name, check_passed, message in structural_checks:
                if not check_passed:
                    error_rows.append({'year': year, 'severity': 'ERROR', 'check': check_name, 'message': message, 'file': str(di_path)})
            mask_grid_matches = False
            if mask_source is not None:
                mask_tags = mask_source.tags()
                mask_image_structure = mask_source.tags(1, ns='IMAGE_STRUCTURE')
                mask_metadata_ok = mask_source.count == 1 and mask_source.dtypes[0] == cfg.MASK_DTYPE and (mask_source.nodata == cfg.MASK_NODATA) and (mask_image_structure.get('NBITS') == str(cfg.MASK_NBITS)) and (mask_tags.get('PROCESS_STEP') == '04valid_mask') and (mask_tags.get('YEAR') == str(year)) and (mask_tags.get('STATISTIC') == cfg.STATISTIC_NAME) and (mask_tags.get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (mask_tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS))) and (mask_tags.get('VALIDITY_SOURCE') == 'knowledge_granulation_classes')
                if not mask_metadata_ok:
                    error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'mask_structure_metadata', 'message': 'Incorrect annual common-mask structure or preprocessing-indicator metadata', 'file': str(mask_path)})
                mask_grid_matches = di_source.width == mask_source.width and di_source.height == mask_source.height and (di_source.crs == mask_source.crs) and np.allclose(np.asarray(tuple(di_source.transform), dtype=np.float64), np.asarray(tuple(mask_source.transform), dtype=np.float64), rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
                if not mask_grid_matches:
                    error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'mask_grid', 'message': 'DSI and annual common-mask grids are not aligned', 'file': str(di_path)})
            valid_pixel_count = 0
            nodata_pixel_count = 0
            mask_valid_pixel_count = 0
            mask_mismatch_pixel_count = 0
            zero_pixel_count = 0
            nan_pixel_count = 0
            inf_pixel_count = 0
            below_zero_count = 0
            above_one_count = 0
            valid_value_blocks = []
            for _, window in di_source.block_windows(1):
                di_data = di_source.read(1, window=window, masked=False)
                if di_source.nodata is None:
                    nodata_mask = di_source.read_masks(1, window=window) == 0
                else:
                    nodata_mask = di_data == di_source.nodata
                di_valid_mask = ~nodata_mask
                candidate_values = di_data[di_valid_mask]
                finite_mask = np.isfinite(candidate_values)
                finite_values = candidate_values[finite_mask]
                valid_pixel_count += int(candidate_values.size)
                nodata_pixel_count += int(nodata_mask.sum())
                nan_pixel_count += int(np.count_nonzero(np.isnan(candidate_values)))
                inf_pixel_count += int(np.count_nonzero(np.isinf(candidate_values)))
                zero_pixel_count += int(np.count_nonzero(finite_values == 0.0))
                below_zero_count += int(np.count_nonzero(finite_values < cfg.DI_VALID_MIN - cfg.VALUE_RANGE_TOLERANCE))
                above_one_count += int(np.count_nonzero(finite_values > cfg.DI_VALID_MAX + cfg.VALUE_RANGE_TOLERANCE))
                if finite_values.size:
                    valid_value_blocks.append(finite_values.astype(np.float32))
                if mask_grid_matches:
                    mask_data = mask_source.read(1, window=window, masked=False)
                    common_valid = mask_data == cfg.MASK_VALID_VALUE
                    mask_valid_pixel_count += int(common_valid.sum())
                    mask_mismatch_pixel_count += int(np.count_nonzero(di_valid_mask != common_valid))
            total_pixel_count = di_source.width * di_source.height
            if valid_value_blocks:
                valid_values = np.concatenate(valid_value_blocks)
                percentiles = np.percentile(valid_values, cfg.DI_QUALITY_PERCENTILES)
                minimum = float(valid_values.min())
                maximum = float(valid_values.max())
                mean_value = float(valid_values.mean(dtype=np.float64))
                std_value = float(valid_values.std(dtype=np.float64, ddof=0))
            else:
                valid_values = np.asarray([], dtype=np.float32)
                percentiles = np.full(5, np.nan, dtype=np.float64)
                minimum = np.nan
                maximum = np.nan
                mean_value = np.nan
                std_value = np.nan
            summary_row.update({'total_pixels': total_pixel_count, 'valid_pixels': valid_pixel_count, 'nodata_pixels': nodata_pixel_count, 'valid_ratio': valid_pixel_count / total_pixel_count if total_pixel_count else np.nan, 'mask_valid_pixels': mask_valid_pixel_count if mask_grid_matches else np.nan, 'mask_mismatch_pixels': mask_mismatch_pixel_count if mask_grid_matches else np.nan, 'zero_pixels': zero_pixel_count, 'nan_pixels': nan_pixel_count, 'inf_pixels': inf_pixel_count, 'below_0_pixels': below_zero_count, 'above_1_pixels': above_one_count, 'minimum': minimum, 'p01': float(percentiles[0]), 'p05': float(percentiles[1]), 'median': float(percentiles[2]), 'p95': float(percentiles[3]), 'p99': float(percentiles[4]), 'maximum': maximum, 'mean': mean_value, 'std': std_value})
            value_checks = [('valid_pixels', valid_pixel_count > 0, 'DSI has zero valid pixels'), ('pixel_accounting', valid_pixel_count + nodata_pixel_count == total_pixel_count, 'Valid and NoData counts do not sum to the total pixel count'), ('nan_pixels', nan_pixel_count == 0, 'DSI valid domain contains NaN'), ('inf_pixels', inf_pixel_count == 0, 'DSI valid domain contains Inf'), ('below_0', below_zero_count == 0, 'DSI contains valid values below zero'), ('above_1', above_one_count == 0, 'DSI contains valid values above one')]
            if mask_grid_matches:
                value_checks.extend([('mask_mismatch', mask_mismatch_pixel_count == 0, 'DSI valid domain differs from the annual common mask'), ('mask_valid_count', valid_pixel_count == mask_valid_pixel_count, 'DSI valid-pixel count differs from mask valid-pixel count')])
            for check_name, check_passed, message in value_checks:
                if not check_passed:
                    error_rows.append({'year': year, 'severity': 'ERROR', 'check': check_name, 'message': message, 'file': str(di_path)})
            expected_period_start = f'{year}-{cfg.PERIOD_START_MONTH:02d}'
            expected_period_end = f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}'
            metadata_checks = [('process_step', tags.get('PROCESS_STEP') == '07calculate_DI', 'Incorrect PROCESS_STEP'), ('year_tag', tags.get('YEAR') == str(year), 'Incorrect YEAR metadata'), ('statistic', tags.get('STATISTIC') == cfg.STATISTIC_NAME, 'Incorrect STATISTIC metadata'), ('period_kind', tags.get('PERIOD_KIND') == cfg.PERIOD_KIND, 'Incorrect PERIOD_KIND metadata'), ('period_start', tags.get('PERIOD_START') == expected_period_start, 'Incorrect PERIOD_START metadata'), ('period_end', tags.get('PERIOD_END') == expected_period_end, 'Incorrect PERIOD_END metadata'), ('indicator_group', tags.get('INDICATOR_GROUP_NAME') == cfg.INDICATOR_GROUP_NAME, 'Incorrect indicator-group metadata'), ('selected_indicators', tags.get('SELECTED_INDICATORS') == ','.join(calculation_indicators), 'Incorrect indicator-order metadata'), ('indicator_count', tags.get('SELECTED_INDICATOR_COUNT') == str(len(calculation_indicators)), 'Incorrect indicator-count metadata'), ('preprocess_indicators', tags.get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS), 'Incorrect common-mask indicator metadata'), ('preprocess_indicator_count', tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS)), 'Incorrect common-mask indicator count'), ('weight_sheet', tags.get('WEIGHT_SHEET') == 'W', 'Weight worksheet metadata is not W'), ('weight_mode', tags.get('WEIGHT_MODE') == weight_mode, 'Incorrect weight-mode metadata'), ('weight_column', tags.get('WEIGHT_COLUMN') == str(weight_column), 'Incorrect weight-column metadata')]
            weight_source_matches = False
            try:
                weight_source_matches = tags.get('WEIGHT_SOURCE') is not None and os.path.normcase(os.path.normpath(tags['WEIGHT_SOURCE'])) == os.path.normcase(os.path.normpath(str(cfg.GK_ER_W_XLSX)))
            except (TypeError, ValueError):
                weight_source_matches = False
            metadata_checks.append(('weight_source', weight_source_matches, 'Incorrect weight-source path metadata'))
            for check_name, check_passed, message in metadata_checks:
                if not check_passed:
                    error_rows.append({'year': year, 'severity': 'ERROR', 'check': check_name, 'message': message, 'file': str(di_path)})
            parsed_weights = []
            parsed_weight_names = []
            weights_parse_ok = True
            try:
                for item in tags.get('APPLIED_WEIGHTS', '').split(';'):
                    name, value = item.split('=', 1)
                    parsed_weight_names.append(name)
                    parsed_weights.append(float(value))
            except (TypeError, ValueError):
                weights_parse_ok = False
            if weights_parse_ok:
                parsed_weights_array = np.asarray(parsed_weights, dtype=np.float64)
                weights_match = parsed_weight_names == list(calculation_indicators) and parsed_weights_array.shape == applied_weights.shape and np.allclose(parsed_weights_array, applied_weights, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE)
            else:
                weights_match = False
            try:
                metadata_weight_sum = float(tags.get('WEIGHT_SUM', np.nan))
            except (TypeError, ValueError):
                metadata_weight_sum = np.nan
            summary_row['weight_sum'] = metadata_weight_sum
            summary_row['weights_match'] = bool(weights_match)
            if not weights_match:
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'applied_weights', 'message': 'Applied-weight metadata differs from the corresponding W column', 'file': str(di_path)})
            if not np.isclose(metadata_weight_sum, 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE):
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'weight_sum', 'message': 'WEIGHT_SUM metadata does not equal one', 'file': str(di_path)})
            clipped_low_total = 0
            clipped_high_total = 0
            clipping_parse_ok = True
            try:
                for item in tags.get('CLIPPED_LOW_COUNTS', '').split(';'):
                    _, value = item.split('=', 1)
                    clipped_low_total += int(value)
                for item in tags.get('CLIPPED_HIGH_COUNTS', '').split(';'):
                    _, value = item.split('=', 1)
                    clipped_high_total += int(value)
            except (TypeError, ValueError):
                clipping_parse_ok = False
            summary_row['clipped_low_pixels'] = clipped_low_total if clipping_parse_ok else np.nan
            summary_row['clipped_high_pixels'] = clipped_high_total if clipping_parse_ok else np.nan
            if not clipping_parse_ok:
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'clipping_metadata', 'message': 'Cannot parse source-value clipping counts', 'file': str(di_path)})
            elif clipped_low_total or clipped_high_total:
                error_rows.append({'year': year, 'severity': 'WARNING', 'check': 'source_clipping', 'message': f'Source values were clipped during DSI calculation; low values {clipped_low_total}; high values: {clipped_high_total}', 'file': str(di_path)})
            metadata_count_match = False
            metadata_stats_match = False
            try:
                metadata_common_count = int(tags.get('COMMON_VALID_PIXEL_COUNT', ''))
                metadata_di_count = int(tags.get('DI_VALID_PIXEL_COUNT', ''))
                metadata_count_match = metadata_common_count == valid_pixel_count and metadata_di_count == valid_pixel_count
            except (TypeError, ValueError):
                metadata_count_match = False
            try:
                metadata_minimum = float(tags.get('DI_MIN', np.nan))
                metadata_maximum = float(tags.get('DI_MAX', np.nan))
                metadata_mean = float(tags.get('DI_MEAN', np.nan))
                metadata_stats_match = np.isclose(metadata_minimum, minimum, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE) and np.isclose(metadata_maximum, maximum, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE) and np.isclose(metadata_mean, mean_value, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE)
            except (TypeError, ValueError):
                metadata_stats_match = False
            summary_row['metadata_count_match'] = bool(metadata_count_match)
            summary_row['metadata_stats_match'] = bool(metadata_stats_match)
            if not metadata_count_match:
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'metadata_pixel_count', 'message': 'Metadata valid-pixel count differs from the recalculated count', 'file': str(di_path)})
            if not metadata_stats_match:
                error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'metadata_statistics', 'message': 'Metadata minimum, maximum, or mean differs from recalculated statistics', 'file': str(di_path)})
    except Exception as error:
        summary_row['open_status'] = 'ERROR'
        error_rows.append({'year': year, 'severity': 'ERROR', 'check': 'open_or_read', 'message': str(error), 'file': str(di_path)})
    year_issues = error_rows[year_error_start:]
    year_error_count = sum((str(item['severity']).upper() == 'ERROR' for item in year_issues))
    year_warning_count = sum((str(item['severity']).upper() == 'WARNING' for item in year_issues))
    summary_row['error_count'] = year_error_count
    summary_row['warning_count'] = year_warning_count
    if year_error_count:
        summary_row['status'] = 'ERROR'
    elif year_warning_count:
        summary_row['status'] = 'WARNING'
    else:
        summary_row['status'] = 'OK'
    summary_rows.append(summary_row)
summary_tables_by_mode = {}
errors_tables_by_mode = {}
for weight_mode in cfg.DI_WEIGHT_MODES:
    summary_table = pd.DataFrame(summary_rows_by_mode[weight_mode])
    errors_table = pd.DataFrame(error_rows_by_mode[weight_mode], columns=error_columns)
    if len(summary_table) != len(cfg.YEARS):
        raise RuntimeError(f'{weight_mode} summary row count is incorrect; required {len(cfg.YEARS)}; actual: {len(summary_table)}')
    if summary_table['year'].astype(int).tolist() != list(cfg.YEARS):
        raise RuntimeError(f'{weight_mode} summary years or ordering are incorrect')
    if not summary_table['weight_mode'].astype(str).eq(weight_mode).all():
        raise RuntimeError(f'{weight_mode} summary weight-mode field is incorrect')
    expected_weight_column_text = [str(year) if weight_mode == 'annual_weights' else cfg.STUDY_PERIOD_SHEET_NAME for year in cfg.YEARS]
    if summary_table['weight_column'].astype(str).tolist() != expected_weight_column_text:
        raise RuntimeError(f'{weight_mode} summary weight-column field is incorrect')
    summary_tables_by_mode[weight_mode] = summary_table
    errors_tables_by_mode[weight_mode] = errors_table
header_fill = PatternFill(fill_type='solid', fgColor='1F4E78')
header_font = Font(color='FFFFFF', bold=True)
status_fill_map = {'OK': PatternFill(fill_type='solid', fgColor='C6EFCE'), 'WARNING': PatternFill(fill_type='solid', fgColor='FFEB9C'), 'ERROR': PatternFill(fill_type='solid', fgColor='FFC7CE')}
temporary_paths_by_mode = {}
workbook_check = None
try:
    for weight_mode in cfg.DI_WEIGHT_MODES:
        summary_table = summary_tables_by_mode[weight_mode]
        errors_table = errors_tables_by_mode[weight_mode]
        formal_output_path = cfg.DI_QUALITY_XLSX_BY_MODE[weight_mode]
        formal_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output_path = formal_output_path.with_name(f'{formal_output_path.stem}.part{formal_output_path.suffix}')
        temporary_output_path.unlink(missing_ok=True)
        temporary_paths_by_mode[weight_mode] = temporary_output_path
        with pd.ExcelWriter(temporary_output_path, engine=cfg.EXCEL_ENGINE) as writer:
            summary_table.to_excel(writer, sheet_name='summary', index=False)
            errors_table.to_excel(writer, sheet_name='errors', index=False)
            for sheet_name in ('summary', 'errors'):
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = 'G2' if sheet_name == 'summary' else 'A2'
                worksheet.sheet_view.showGridLines = False
                worksheet.auto_filter.ref = worksheet.dimensions
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center')
                for column_cells in worksheet.columns:
                    column_letter = column_cells[0].column_letter
                    header = str(column_cells[0].value)
                    maximum_length = max((len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells))
                    width_cap = 60 if header in {'file', 'message'} else 32
                    worksheet.column_dimensions[column_letter].width = min(max(maximum_length + 2, 12), width_cap)
            summary_worksheet = writer.sheets['summary']
            summary_headers = {cell.value: cell.column for cell in summary_worksheet[1]}
            count_headers = {'total_pixels', 'valid_pixels', 'nodata_pixels', 'mask_valid_pixels', 'mask_mismatch_pixels', 'zero_pixels', 'nan_pixels', 'inf_pixels', 'below_0_pixels', 'above_1_pixels', 'clipped_low_pixels', 'clipped_high_pixels', 'error_count', 'warning_count'}
            statistic_headers = {'minimum', 'p01', 'p05', 'median', 'p95', 'p99', 'maximum', 'mean', 'std', 'weight_sum'}
            for header in count_headers:
                if header in summary_headers:
                    column_index = summary_headers[header]
                    for row_index in range(2, len(summary_table) + 2):
                        summary_worksheet.cell(row_index, column_index).number_format = '#,##0'
            for header in statistic_headers:
                if header in summary_headers:
                    column_index = summary_headers[header]
                    for row_index in range(2, len(summary_table) + 2):
                        summary_worksheet.cell(row_index, column_index).number_format = '0.000000000000'
            valid_ratio_column = summary_headers['valid_ratio']
            for row_index in range(2, len(summary_table) + 2):
                summary_worksheet.cell(row_index, valid_ratio_column).number_format = '0.0000%'
            status_column = summary_headers['status']
            for row_index in range(2, len(summary_table) + 2):
                status_cell = summary_worksheet.cell(row_index, status_column)
                if status_cell.value in status_fill_map:
                    status_cell.fill = status_fill_map[status_cell.value]
                    status_cell.font = Font(bold=True)
            errors_worksheet = writer.sheets['errors']
            if len(errors_table):
                severity_column = error_columns.index('severity') + 1
                for row_index in range(2, len(errors_table) + 2):
                    severity_cell = errors_worksheet.cell(row_index, severity_column)
                    if severity_cell.value in status_fill_map:
                        severity_cell.fill = status_fill_map[severity_cell.value]
                        severity_cell.font = Font(bold=True)
        workbook_check = load_workbook(temporary_output_path, read_only=True, data_only=True)
        if workbook_check.sheetnames != ['summary', 'errors']:
            raise RuntimeError(f'{weight_mode} quality workbook sheet names or order are incorrect: {workbook_check.sheetnames}')
        summary_check = workbook_check['summary']
        errors_check = workbook_check['errors']
        summary_headers_check = [summary_check.cell(1, column).value for column in range(1, len(summary_table.columns) + 1)]
        errors_headers_check = [errors_check.cell(1, column).value for column in range(1, len(error_columns) + 1)]
        if summary_headers_check != list(summary_table.columns):
            raise RuntimeError(f'{weight_mode} summary headers were written incorrectly')
        if errors_headers_check != error_columns:
            raise RuntimeError(f'{weight_mode} errors headers were written incorrectly')
        if summary_check.max_row - 1 != len(cfg.YEARS):
            raise RuntimeError(f'{weight_mode} summary annual row count is incorrect')
        if errors_check.max_row - 1 != len(errors_table):
            raise RuntimeError(f'{weight_mode} errors issue-row count is incorrect')
        year_column_index = list(summary_table.columns).index('year') + 1
        year_column_values = [summary_check.cell(row, year_column_index).value for row in range(2, len(cfg.YEARS) + 2)]
        if year_column_values != list(cfg.YEARS):
            raise RuntimeError(f'{weight_mode} summary years or ordering are incorrect')
        workbook_check.close()
        workbook_check = None
    for weight_mode in cfg.DI_WEIGHT_MODES:
        os.replace(temporary_paths_by_mode[weight_mode], cfg.DI_QUALITY_XLSX_BY_MODE[weight_mode])
except Exception:
    if workbook_check is not None:
        workbook_check.close()
    for temporary_output_path in temporary_paths_by_mode.values():
        temporary_output_path.unlink(missing_ok=True)
    raise
for weight_mode in cfg.DI_WEIGHT_MODES:
    summary_table = summary_tables_by_mode[weight_mode]
    errors_table = errors_tables_by_mode[weight_mode]
    status_counts = summary_table['status'].value_counts().to_dict()
    global_issue_count = int(errors_table['year'].isna().sum()) if len(errors_table) else 0
    print(f"{weight_mode} DSI quality checks completed; OK={status_counts.get('OK', 0)}，WARNING={status_counts.get('WARNING', 0)}，ERROR={status_counts.get('ERROR', 0)}; global issues={global_issue_count}。")
    print(f'DSI quality workbook: {cfg.DI_QUALITY_XLSX_BY_MODE[weight_mode]}')
