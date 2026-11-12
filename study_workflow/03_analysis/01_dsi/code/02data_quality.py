"""Check input raster structure, validity, scaling, and metadata."""
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
from rasterio.errors import RasterioError
from tqdm import tqdm
import config as cfg
expected_jobs = []
issues = []
inventory_rows = []
profile_rows = []
value_rows = []
reference_grid = None
output_path = cfg.DATA_QUALITY_XLSX
temporary_path = output_path.with_name(f'{output_path.stem}.part{output_path.suffix}')
if output_path.exists() and (not cfg.OVERWRITE):
    raise FileExistsError(f'Quality workbook exists and OVERWRITE=False: {output_path}')
for indicator in cfg.PREPROCESS_INDICATORS:
    indicator_dir = cfg.INDICATOR_CONFIG[indicator]['input_dir']
    if not indicator_dir.is_dir():
        issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': None, 'file': str(indicator_dir), 'check': 'indicator_directory', 'message': 'Indicator input directory does not exist'})
    for year in cfg.YEARS:
        expected_name = cfg.INPUT_FILE_PATTERN.format(indicator=indicator, year=year)
        expected_jobs.append({'indicator': indicator, 'year': year, 'path': indicator_dir / expected_name})
for job in tqdm(expected_jobs, desc='Check input rasters', unit='file', dynamic_ncols=True):
    indicator = job['indicator']
    year = job['year']
    source_path = job['path']
    issue_start = len(issues)
    inventory_row = {'indicator': indicator, 'year': year, 'expected_file': source_path.name, 'path': str(source_path), 'exists': source_path.is_file(), 'open_status': 'NOT_CHECKED', 'error_count': 0, 'warning_count': 0}
    if not source_path.is_file():
        inventory_row['open_status'] = 'MISSING'
        issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'file_exists', 'message': 'Annual input file does not exist'})
        inventory_row['error_count'] = 1
        inventory_rows.append(inventory_row)
        continue
    try:
        with rasterio.open(source_path) as source:
            inventory_row['open_status'] = 'OPENED'
            tags = source.tags()
            image_structure_tags = source.tags(ns='IMAGE_STRUCTURE')
            crs_text = source.crs.to_string() if source.crs else None
            transform_values = tuple(source.transform)
            block_shape = source.block_shapes[0] if source.block_shapes else (None, None)
            compression = source.compression.value if source.compression else None
            scale = source.scales[0] if source.scales else None
            offset = source.offsets[0] if source.offsets else None
            expected_period_start = f'{year}-{cfg.PERIOD_START_MONTH:02d}'
            expected_period_end = f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}'
            if reference_grid is None:
                reference_grid = {'width': source.width, 'height': source.height, 'crs': source.crs, 'transform': transform_values}
            if source.driver != 'GTiff':
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'driver', 'message': f'Required driver GTiff; actual: {source.driver}'})
            if source.count != cfg.INPUT_BAND_COUNT:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'band_count', 'message': f'Required: {cfg.INPUT_BAND_COUNT} bands; actual: {source.count}'})
            if source.dtypes[0] != cfg.INPUT_DTYPE:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'dtype', 'message': f'Required: {cfg.INPUT_DTYPE}; actual value: {source.dtypes[0]}'})
            nodata_matches = source.nodata is not None and np.isclose(source.nodata, cfg.INPUT_NODATA, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE)
            if not nodata_matches:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'nodata', 'message': f'Required NoData={cfg.INPUT_NODATA}; actual value: {source.nodata}'})
            if source.width != cfg.EXPECTED_WIDTH or source.height != cfg.EXPECTED_HEIGHT:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'raster_size', 'message': f'Required: {cfg.EXPECTED_WIDTH}×{cfg.EXPECTED_HEIGHT}; actual value: {source.width}×{source.height}'})
            if crs_text != cfg.EXPECTED_CRS:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'crs', 'message': f'Required: {cfg.EXPECTED_CRS}; actual value: {crs_text}'})
            x_resolution_matches = np.isclose(abs(source.res[0]), cfg.EXPECTED_X_RESOLUTION, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
            y_resolution_matches = np.isclose(abs(source.res[1]), cfg.EXPECTED_Y_RESOLUTION, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
            if not x_resolution_matches or not y_resolution_matches:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'resolution', 'message': f'Required resolution: {cfg.EXPECTED_X_RESOLUTION}×{cfg.EXPECTED_Y_RESOLUTION}; actual value: {source.res}'})
            reference_transform_matches = np.allclose(np.asarray(transform_values, dtype=np.float64), np.asarray(reference_grid['transform'], dtype=np.float64), rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
            reference_crs_matches = source.crs == reference_grid['crs']
            reference_size_matches = source.width == reference_grid['width'] and source.height == reference_grid['height']
            if not (reference_transform_matches and reference_crs_matches and reference_size_matches):
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'common_grid', 'message': 'Dimensions, CRS, or affine transform differ from the first successfully opened raster'})
            if not source.is_tiled:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'tiled', 'message': 'Input GeoTIFF is not tiled'})
            if block_shape != (cfg.GTIFF_BLOCK_Y_SIZE, cfg.GTIFF_BLOCK_X_SIZE):
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'block_shape', 'message': f'Required tile size: {cfg.GTIFF_BLOCK_X_SIZE}×{cfg.GTIFF_BLOCK_Y_SIZE}; actual value: {block_shape[1]}×{block_shape[0]}'})
            if compression != cfg.GTIFF_COMPRESSION.upper():
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'compression', 'message': f'Required: {cfg.GTIFF_COMPRESSION.upper()}; actual value: {compression}'})
            scale_matches = scale is not None and np.isclose(scale, cfg.INPUT_SCALE, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE)
            offset_matches = offset is not None and np.isclose(offset, cfg.INPUT_OFFSET, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE)
            if not scale_matches or not offset_matches:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'scale_offset', 'message': f'Required scale={cfg.INPUT_SCALE}、offset={cfg.INPUT_OFFSET}; actual scale={scale}、offset={offset}'})
            expected_tag_values = {'STATISTIC': cfg.STATISTIC_NAME, 'PERIOD_KIND': cfg.PERIOD_KIND, 'PERIOD_DEFINITION': cfg.PERIOD_DEFINITION, 'PERIOD_START': expected_period_start, 'PERIOD_END': expected_period_end, 'SOURCE_COUNT': str(cfg.EXPECTED_SOURCE_COUNT)}
            for tag_name, expected_tag_value in expected_tag_values.items():
                actual_tag_value = tags.get(tag_name)
                if actual_tag_value != expected_tag_value:
                    issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': f'metadata_{tag_name.lower()}', 'message': f'Required: {tag_name}={expected_tag_value}; actual value: {actual_tag_value}'})
            if not tags.get('NODATA_POLICY'):
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'metadata_nodata_policy', 'message': 'Missing NODATA_POLICY metadata'})
            total_pixel_count = source.width * source.height
            valid_pixel_count = 0
            masked_pixel_count = 0
            nodata_value_count = 0
            nan_pixel_count = 0
            inf_pixel_count = 0
            below_range_count = 0
            above_range_count = 0
            zero_value_count = 0
            scale_max_value_count = 0
            valid_minimum = np.inf
            valid_maximum = -np.inf
            for _, window in source.block_windows(1):
                data = source.read(1, window=window, masked=False)
                data_mask = source.read_masks(1, window=window) > 0
                finite_mask = np.isfinite(data)
                configured_nodata_mask = data == cfg.INPUT_NODATA
                valid_mask = data_mask & finite_mask & ~configured_nodata_mask
                valid_values = data[valid_mask]
                valid_pixel_count += int(valid_values.size)
                masked_pixel_count += int(np.count_nonzero(~data_mask))
                nodata_value_count += int(np.count_nonzero(configured_nodata_mask))
                nan_pixel_count += int(np.count_nonzero(data_mask & np.isnan(data)))
                inf_pixel_count += int(np.count_nonzero(data_mask & np.isinf(data)))
                if valid_values.size:
                    valid_minimum = min(valid_minimum, float(valid_values.min()))
                    valid_maximum = max(valid_maximum, float(valid_values.max()))
                    below_range_count += int(np.count_nonzero(valid_values < cfg.NORMALIZED_MIN))
                    above_range_count += int(np.count_nonzero(valid_values > cfg.INPUT_SCALE_MAX))
                    zero_value_count += int(np.count_nonzero(valid_values == 0))
                    scale_max_value_count += int(np.count_nonzero(valid_values == cfg.INPUT_SCALE_MAX))
            invalid_pixel_count = total_pixel_count - valid_pixel_count
            valid_minimum_output = valid_minimum if np.isfinite(valid_minimum) else None
            valid_maximum_output = valid_maximum if np.isfinite(valid_maximum) else None
            if valid_pixel_count == 0:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'valid_pixel_count', 'message': 'No valid pixels'})
            if nan_pixel_count or inf_pixel_count:
                issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'nan_inf', 'message': f'NaN within valid mask: {nan_pixel_count}，Inf={inf_pixel_count}'})
            if below_range_count or above_range_count:
                range_severity = 'WARNING' if cfg.CLIP_NORMALIZED_VALUES else 'ERROR'
                issues.append({'severity': range_severity, 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'valid_value_range', 'message': f'Valid pixels below zero: {below_range_count}; above {cfg.INPUT_SCALE_MAX}: valid pixels {above_range_count}'})
            profile_rows.append({'indicator': indicator, 'year': year, 'file': source_path.name, 'driver': source.driver, 'band_count': source.count, 'width': source.width, 'height': source.height, 'dtype': source.dtypes[0], 'nodata': source.nodata, 'crs': crs_text, 'x_resolution': source.res[0], 'y_resolution': source.res[1], 'transform_a': source.transform.a, 'transform_b': source.transform.b, 'transform_c': source.transform.c, 'transform_d': source.transform.d, 'transform_e': source.transform.e, 'transform_f': source.transform.f, 'block_x_size': block_shape[1], 'block_y_size': block_shape[0], 'is_tiled': source.is_tiled, 'compression': compression, 'predictor': image_structure_tags.get('PREDICTOR'), 'scale': scale, 'offset': offset, 'statistic': tags.get('STATISTIC'), 'period_kind': tags.get('PERIOD_KIND'), 'period_definition': tags.get('PERIOD_DEFINITION'), 'period_start': tags.get('PERIOD_START'), 'period_end': tags.get('PERIOD_END'), 'source_count': tags.get('SOURCE_COUNT'), 'nodata_policy': tags.get('NODATA_POLICY')})
            value_rows.append({'indicator': indicator, 'year': year, 'file': source_path.name, 'total_pixels': total_pixel_count, 'valid_pixels': valid_pixel_count, 'invalid_pixels': invalid_pixel_count, 'masked_pixels': masked_pixel_count, 'nodata_value_pixels': nodata_value_count, 'nan_pixels_in_data_mask': nan_pixel_count, 'inf_pixels_in_data_mask': inf_pixel_count, 'valid_min': valid_minimum_output, 'valid_max': valid_maximum_output, 'below_zero_pixels': below_range_count, 'above_scale_max_pixels': above_range_count, 'zero_value_pixels': zero_value_count, 'scale_max_value_pixels': scale_max_value_count, 'clip_required': bool(below_range_count or above_range_count)})
    except (RasterioError, OSError, ValueError) as error:
        inventory_row['open_status'] = 'OPEN_ERROR'
        issues.append({'severity': 'ERROR', 'indicator': indicator, 'year': year, 'file': str(source_path), 'check': 'open_or_read', 'message': str(error)})
    current_issues = issues[issue_start:]
    inventory_row['error_count'] = sum((item['severity'] == 'ERROR' for item in current_issues))
    inventory_row['warning_count'] = sum((item['severity'] == 'WARNING' for item in current_issues))
    if inventory_row['open_status'] == 'OPENED':
        inventory_row['open_status'] = 'ERROR' if inventory_row['error_count'] else 'OK'
    inventory_rows.append(inventory_row)
inventory_columns = ['indicator', 'year', 'expected_file', 'path', 'exists', 'open_status', 'error_count', 'warning_count']
profile_columns = ['indicator', 'year', 'file', 'driver', 'band_count', 'width', 'height', 'dtype', 'nodata', 'crs', 'x_resolution', 'y_resolution', 'transform_a', 'transform_b', 'transform_c', 'transform_d', 'transform_e', 'transform_f', 'block_x_size', 'block_y_size', 'is_tiled', 'compression', 'predictor', 'scale', 'offset', 'statistic', 'period_kind', 'period_definition', 'period_start', 'period_end', 'source_count', 'nodata_policy']
value_columns = ['indicator', 'year', 'file', 'total_pixels', 'valid_pixels', 'invalid_pixels', 'masked_pixels', 'nodata_value_pixels', 'nan_pixels_in_data_mask', 'inf_pixels_in_data_mask', 'valid_min', 'valid_max', 'below_zero_pixels', 'above_scale_max_pixels', 'zero_value_pixels', 'scale_max_value_pixels', 'clip_required']
issue_columns = ['severity', 'indicator', 'year', 'file', 'check', 'message']
inventory_table = pd.DataFrame(inventory_rows, columns=inventory_columns)
profile_table = pd.DataFrame(profile_rows, columns=profile_columns)
value_table = pd.DataFrame(value_rows, columns=value_columns)
issue_table = pd.DataFrame(issues, columns=issue_columns)
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path.unlink(missing_ok=True)
tables = {'file_inventory': inventory_table, 'raster_profile': profile_table, 'value_range': value_table, 'errors': issue_table}
try:
    with pd.ExcelWriter(temporary_path, engine=cfg.EXCEL_ENGINE) as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = 'A2'
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                column_letter = column_cells[0].column_letter
                maximum_length = max((len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells))
                worksheet.column_dimensions[column_letter].width = min(maximum_length + 2, 60)
    workbook_check = load_workbook(temporary_path, read_only=True, data_only=True)
    expected_sheets = list(tables)
    actual_sheets = workbook_check.sheetnames
    workbook_check.close()
    if actual_sheets != expected_sheets:
        raise RuntimeError(f'Incomplete quality workbook; required sheets: {expected_sheets}; actual: {actual_sheets}')
    os.replace(temporary_path, output_path)
except Exception:
    temporary_path.unlink(missing_ok=True)
    raise
error_count = int((issue_table['severity'] == 'ERROR').sum())
warning_count = int((issue_table['severity'] == 'WARNING').sum())
print(f'Expected files: {len(expected_jobs)}')
print(f'Preprocessing indicator count: {len(cfg.PREPROCESS_INDICATORS)}')
print(f'Structural records created: {len(profile_table)}')
print(f'ERROR count: {error_count}')
print(f'WARNING count: {warning_count}')
print(f'Quality workbook: {output_path}')
if error_count:
    raise RuntimeError(f'Data-quality checks found {error_count} errors; see {output_path} worksheet errors. Do not proceed to step 3.')
print('Data-quality checks passed; proceed to step 3: knowledge granulation.')
