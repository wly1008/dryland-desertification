"""Build annual common-valid masks across preprocessing indicators."""
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
from tqdm import tqdm
import config as cfg
if not cfg.DATA_QUALITY_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 2 data-quality workbook: {cfg.DATA_QUALITY_XLSX}')
quality_errors = pd.read_excel(cfg.DATA_QUALITY_XLSX, sheet_name='errors', engine=cfg.EXCEL_ENGINE)
quality_inventory = pd.read_excel(cfg.DATA_QUALITY_XLSX, sheet_name='file_inventory', engine=cfg.EXCEL_ENGINE)
if not quality_errors.empty:
    error_rows = quality_errors[quality_errors['severity'].astype(str).str.upper() == 'ERROR']
    if not error_rows.empty:
        raise RuntimeError(f'Step 2 still has {len(error_rows)} errors prevent common-valid mask calculation')
expected_quality_rows = len(cfg.PREPROCESS_INDICATORS) * len(cfg.YEARS)
inventory_status_ok = quality_inventory['exists'].astype(bool) & quality_inventory['open_status'].astype(str).eq('OK') & quality_inventory['error_count'].fillna(0).astype(int).eq(0)
if len(quality_inventory) != expected_quality_rows or not bool(inventory_status_ok.all()):
    raise RuntimeError('Step 2 manifest differs from configuration or contains failed records')
if set(quality_inventory['indicator'].astype(str)) != set(cfg.PREPROCESS_INDICATORS):
    raise RuntimeError('Step 2 indicators do not match the complete preprocessing indicator set')
if set(quality_inventory['year'].astype(int)) != set(cfg.YEARS):
    raise RuntimeError('Step 2 years do not match the current configuration')
inventory_pairs = list(zip(quality_inventory['indicator'].astype(str), quality_inventory['year'].astype(int)))
expected_inventory_pairs = {(indicator, year) for indicator in cfg.PREPROCESS_INDICATORS for year in cfg.YEARS}
if len(inventory_pairs) != len(set(inventory_pairs)):
    raise RuntimeError('Step 2 file manifest contains duplicate indicator-year records')
if set(inventory_pairs) != expected_inventory_pairs:
    raise RuntimeError('Step 2 file manifest is missing expected indicator-year combinations')
granulation_paths = []
for indicator in cfg.PREPROCESS_INDICATORS:
    for year in cfg.YEARS:
        granulation_paths.append(cfg.GRANULATION_OUTPUT_DIR / indicator / cfg.GRANULATION_FILE_PATTERN.format(indicator=indicator, year=year))
missing_granulation_paths = [path for path in granulation_paths if not path.is_file()]
if missing_granulation_paths:
    missing_preview = '\n'.join((str(path) for path in missing_granulation_paths[:20]))
    raise FileNotFoundError(f'Step 3 is missing {len(missing_granulation_paths)} reclassified rasters:\n{missing_preview}')
for granulation_path in tqdm(granulation_paths, desc='Check step 3 outputs', unit='file', dynamic_ncols=True):
    with rasterio.open(granulation_path) as granulation:
        tags = granulation.tags()
        expected_indicator = granulation_path.parent.name
        expected_year = granulation_path.stem.rsplit('_', 1)[-1]
        granulation_ok = granulation.count == 1 and granulation.dtypes[0] == cfg.RECLASS_DTYPE and (granulation.nodata == cfg.RECLASS_NODATA) and (tags.get('PROCESS_STEP') == '03knowledge_granulation') and (tags.get('INDICATOR') == expected_indicator) and (tags.get('YEAR') == expected_year) and (tags.get('BIN_COUNT') == str(cfg.BIN_COUNT)) and (tags.get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS)))
        if not granulation_ok:
            raise RuntimeError(f'Step 3 reclassified raster failed structure or metadata checks: {granulation_path}')
jobs = []
for year in cfg.YEARS:
    input_paths = {indicator: cfg.GRANULATION_OUTPUT_DIR / indicator / cfg.GRANULATION_FILE_PATTERN.format(indicator=indicator, year=year) for indicator in cfg.PREPROCESS_INDICATORS}
    jobs.append({'year': year, 'input_paths': input_paths, 'output_path': cfg.VALID_MASK_OUTPUT_DIR / cfg.MASK_FILE_PATTERN.format(year=year)})
missing_inputs = [path for job in jobs for path in job['input_paths'].values() if not path.is_file()]
if missing_inputs:
    missing_preview = '\n'.join((str(path) for path in missing_inputs[:20]))
    raise FileNotFoundError(f'Common-valid masks are missing {len(missing_inputs)} reclassified rasters:\n{missing_preview}')
existing_outputs = [job['output_path'] for job in jobs if job['output_path'].exists()]
if cfg.VALID_PIXEL_COUNT_XLSX.exists():
    existing_outputs.append(cfg.VALID_PIXEL_COUNT_XLSX)
if existing_outputs and (not cfg.OVERWRITE):
    existing_preview = '\n'.join((str(path) for path in existing_outputs[:20]))
    raise FileExistsError(f'Found {len(existing_outputs)} step 4 outputs already exist and OVERWRITE=False:\n{existing_preview}')
summary_rows = []
written_mask_count = 0
for job in tqdm(jobs, desc='Calculate common-valid masks', unit='year', dynamic_ncols=True):
    year = job['year']
    input_paths = job['input_paths']
    output_path = job['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.stem}.part{output_path.suffix}')
    temporary_path.unlink(missing_ok=True)
    indicator_valid_counts = {indicator: 0 for indicator in cfg.PREPROCESS_INDICATORS}
    common_valid_count = 0
    try:
        with ExitStack() as stack:
            sources = {indicator: stack.enter_context(rasterio.open(path)) for indicator, path in input_paths.items()}
            reference_indicator = cfg.PREPROCESS_INDICATORS[0]
            reference = sources[reference_indicator]
            reference_transform = np.asarray(tuple(reference.transform), dtype=np.float64)
            for indicator, source in sources.items():
                transform_matches = np.allclose(np.asarray(tuple(source.transform), dtype=np.float64), reference_transform, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
                source_ok = source.count == 1 and source.width == reference.width and (source.height == reference.height) and (source.crs == reference.crs) and transform_matches and (source.dtypes[0] == cfg.RECLASS_DTYPE) and (source.nodata == cfg.RECLASS_NODATA) and (source.tags().get('PROCESS_STEP') == '03knowledge_granulation') and (source.tags().get('INDICATOR') == indicator) and (source.tags().get('YEAR') == str(year))
                if not source_ok:
                    raise RuntimeError(f'Annual reclassified raster structure, metadata, or grid mismatch: {indicator} {year}')
            output_profile = reference.profile.copy()
            output_profile.pop('predictor', None)
            output_profile.update(driver='GTiff', count=1, dtype=cfg.MASK_DTYPE, nodata=cfg.MASK_NODATA, nbits=cfg.MASK_NBITS, tiled=cfg.GTIFF_TILED, blockxsize=cfg.GTIFF_BLOCK_X_SIZE, blockysize=cfg.GTIFF_BLOCK_Y_SIZE, compress=cfg.GTIFF_COMPRESSION, zlevel=cfg.GTIFF_ZLEVEL, bigtiff=cfg.GTIFF_BIGTIFF)
            if cfg.MASK_PREDICTOR is not None:
                output_profile['predictor'] = cfg.MASK_PREDICTOR
            with rasterio.open(temporary_path, 'w', **output_profile) as destination:
                for _, window in reference.block_windows(1):
                    common_valid = np.ones((int(window.height), int(window.width)), dtype=bool)
                    for indicator, source in sources.items():
                        source_data = source.read(1, window=window, masked=False)
                        source_mask = source.read_masks(1, window=window) > 0
                        indicator_valid = source_mask & (source_data >= cfg.RECLASS_MIN_VALUE) & (source_data <= cfg.RECLASS_MAX_VALUE) & (source_data != cfg.RECLASS_NODATA)
                        indicator_valid_counts[indicator] += int(indicator_valid.sum())
                        common_valid &= indicator_valid
                    mask_data = np.full(common_valid.shape, cfg.MASK_INVALID_VALUE, dtype=np.uint8)
                    mask_data[common_valid] = cfg.MASK_VALID_VALUE
                    common_valid_count += int(common_valid.sum())
                    destination.write(mask_data, 1, window=window)
                total_pixel_count = reference.width * reference.height
                common_invalid_count = total_pixel_count - common_valid_count
                destination.set_band_description(1, f'common_valid_mask_{year}')
                destination.update_tags(PROCESS_STEP='04valid_mask', YEAR=str(year), STATISTIC=cfg.STATISTIC_NAME, PERIOD_KIND=cfg.PERIOD_KIND, PERIOD_START=f'{year}-{cfg.PERIOD_START_MONTH:02d}', PERIOD_END=f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}', PREPROCESS_INDICATORS=','.join(cfg.PREPROCESS_INDICATORS), PREPROCESS_INDICATOR_COUNT=str(len(cfg.PREPROCESS_INDICATORS)), VALIDITY_SOURCE='knowledge_granulation_classes', VALID_CLASS_MIN=str(cfg.RECLASS_MIN_VALUE), VALID_CLASS_MAX=str(cfg.RECLASS_MAX_VALUE), MASK_ENCODING='valid=1; invalid_and_nodata=0', MASK_NBITS=str(cfg.MASK_NBITS), OUTPUT_DTYPE=cfg.MASK_DTYPE, OUTPUT_NODATA=str(cfg.MASK_NODATA), TOTAL_PIXEL_COUNT=str(total_pixel_count), COMMON_VALID_PIXEL_COUNT=str(common_valid_count), COMMON_INVALID_PIXEL_COUNT=str(common_invalid_count))
        with rasterio.open(temporary_path) as check:
            image_structure = check.tags(1, ns='IMAGE_STRUCTURE')
            structural_check = check.count == 1 and check.width == output_profile['width'] and (check.height == output_profile['height']) and (check.dtypes[0] == cfg.MASK_DTYPE) and (check.nodata == cfg.MASK_NODATA) and (check.crs == output_profile['crs']) and check.transform.almost_equals(output_profile['transform']) and (image_structure.get('NBITS') == str(cfg.MASK_NBITS)) and (check.tags().get('PROCESS_STEP') == '04valid_mask') and (check.tags().get('YEAR') == str(year)) and (check.tags().get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (check.tags().get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS))) and (check.tags().get('VALIDITY_SOURCE') == 'knowledge_granulation_classes')
            if not structural_check:
                raise RuntimeError('Temporary mask failed structure, NBITS, or metadata checks')
            check_valid_count = 0
            invalid_code_count = 0
            for _, window in check.block_windows(1):
                check_data = check.read(1, window=window, masked=False)
                check_valid_count += int(np.count_nonzero(check_data == cfg.MASK_VALID_VALUE))
                invalid_code_count += int(np.count_nonzero((check_data != cfg.MASK_INVALID_VALUE) & (check_data != cfg.MASK_VALID_VALUE)))
            if check_valid_count != common_valid_count:
                raise RuntimeError(f'Temporary mask valid-pixel count mismatch; calculated {common_valid_count}; verified {check_valid_count}')
            if invalid_code_count:
                raise RuntimeError(f'Temporary mask contains {invalid_code_count} invalid codes')
        os.replace(temporary_path, output_path)
        written_mask_count += 1
        total_pixel_count = output_profile['width'] * output_profile['height']
        summary_row = {'year': year, 'total_pixels': total_pixel_count, 'valid_pixels': common_valid_count, 'invalid_pixels': total_pixel_count - common_valid_count, 'valid_ratio': common_valid_count / total_pixel_count, 'indicator_count': len(cfg.PREPROCESS_INDICATORS), 'indicators': ','.join(cfg.PREPROCESS_INDICATORS), 'statistic': cfg.STATISTIC_NAME, 'period_start': f'{year}-{cfg.PERIOD_START_MONTH:02d}', 'period_end': f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}', 'mask_file': output_path.name}
        for indicator in cfg.PREPROCESS_INDICATORS:
            summary_row[f'{indicator}_valid_pixels'] = indicator_valid_counts[indicator]
        summary_rows.append(summary_row)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f'Common-valid mask calculation failed: {year}\nOutput: {output_path}\n{error}') from error
summary_table = pd.DataFrame(summary_rows)
excel_output_path = cfg.VALID_PIXEL_COUNT_XLSX
excel_output_path.parent.mkdir(parents=True, exist_ok=True)
excel_temporary_path = excel_output_path.with_name(f'{excel_output_path.stem}.part{excel_output_path.suffix}')
excel_temporary_path.unlink(missing_ok=True)
try:
    with pd.ExcelWriter(excel_temporary_path, engine=cfg.EXCEL_ENGINE) as writer:
        summary_table.to_excel(writer, sheet_name='valid_pixel_count', index=False)
        worksheet = writer.sheets['valid_pixel_count']
        worksheet.freeze_panes = 'A2'
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.column_dimensions['D'].width = 16
        worksheet.column_dimensions['E'].width = 14
        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            maximum_length = max((len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells))
            worksheet.column_dimensions[column_letter].width = min(max(worksheet.column_dimensions[column_letter].width, maximum_length + 2), 60)
        for cell in worksheet['E'][1:]:
            cell.number_format = '0.0000%'
    workbook_check = load_workbook(excel_temporary_path, read_only=True, data_only=True)
    worksheet_check = workbook_check['valid_pixel_count']
    excel_data_row_count = worksheet_check.max_row - 1
    workbook_check.close()
    if excel_data_row_count != len(cfg.YEARS):
        raise RuntimeError(f'Incorrect valid-pixel summary row count; required {len(cfg.YEARS)}; actual: {excel_data_row_count}')
    os.replace(excel_temporary_path, excel_output_path)
except Exception:
    excel_temporary_path.unlink(missing_ok=True)
    raise
print(f'Common-valid masks completed; wrote {written_mask_count} rasters.')
print(f'Indicators used for the common-valid domain: {len(cfg.PREPROCESS_INDICATORS)}')
print(f'Mask directory: {cfg.VALID_MASK_OUTPUT_DIR}')
print(f'Valid-pixel summary: {cfg.VALID_PIXEL_COUNT_XLSX}')
