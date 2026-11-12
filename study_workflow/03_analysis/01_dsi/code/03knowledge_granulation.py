"""Assign normalized indicator values to knowledge-granulation classes."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import config as cfg
if not cfg.DATA_QUALITY_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 2 data-quality workbook: {cfg.DATA_QUALITY_XLSX}')
quality_errors = pd.read_excel(cfg.DATA_QUALITY_XLSX, sheet_name='errors', engine=cfg.EXCEL_ENGINE)
quality_inventory = pd.read_excel(cfg.DATA_QUALITY_XLSX, sheet_name='file_inventory', engine=cfg.EXCEL_ENGINE)
if not quality_errors.empty:
    error_rows = quality_errors[quality_errors['severity'].astype(str).str.upper() == 'ERROR']
    if not error_rows.empty:
        raise RuntimeError(f'Step 2 still has {len(error_rows)} errors prevent knowledge granulation: {cfg.DATA_QUALITY_XLSX}')
expected_quality_rows = len(cfg.PREPROCESS_INDICATORS) * len(cfg.YEARS)
if len(quality_inventory) != expected_quality_rows:
    raise RuntimeError(f'Incorrect step 2 manifest row count; required {expected_quality_rows}; actual: {len(quality_inventory)}')
inventory_indicators = set(quality_inventory['indicator'].astype(str))
inventory_years = set(quality_inventory['year'].astype(int))
if inventory_indicators != set(cfg.PREPROCESS_INDICATORS):
    raise RuntimeError('Step 2 manifest indicators do not match PREPROCESS_INDICATORS')
if inventory_years != set(cfg.YEARS):
    raise RuntimeError('Step 2 manifest years do not match YEARS')
inventory_pairs = list(zip(quality_inventory['indicator'].astype(str), quality_inventory['year'].astype(int)))
expected_inventory_pairs = {(indicator, year) for indicator in cfg.PREPROCESS_INDICATORS for year in cfg.YEARS}
if len(inventory_pairs) != len(set(inventory_pairs)):
    raise RuntimeError('Step 2 file manifest contains duplicate indicator-year records')
if set(inventory_pairs) != expected_inventory_pairs:
    raise RuntimeError('Step 2 file manifest is missing expected indicator-year combinations')
inventory_status_ok = quality_inventory['exists'].astype(bool) & quality_inventory['open_status'].astype(str).eq('OK') & quality_inventory['error_count'].fillna(0).astype(int).eq(0)
if not bool(inventory_status_ok.all()):
    invalid_inventory = quality_inventory.loc[~inventory_status_ok, ['indicator', 'year', 'path', 'open_status', 'error_count']]
    raise RuntimeError(f'Step 2 contains failed records; knowledge granulation cannot proceed:\n{invalid_inventory.to_string(index=False)}')
jobs = []
for indicator in cfg.PREPROCESS_INDICATORS:
    input_dir = cfg.INDICATOR_CONFIG[indicator]['input_dir']
    output_dir = cfg.GRANULATION_OUTPUT_DIR / indicator
    for year in cfg.YEARS:
        input_name = cfg.INPUT_FILE_PATTERN.format(indicator=indicator, year=year)
        output_name = cfg.GRANULATION_FILE_PATTERN.format(indicator=indicator, year=year)
        jobs.append({'indicator': indicator, 'year': year, 'input_path': input_dir / input_name, 'output_path': output_dir / output_name})
missing_inputs = [job['input_path'] for job in jobs if not job['input_path'].is_file()]
if missing_inputs:
    missing_preview = '\n'.join((str(path) for path in missing_inputs[:20]))
    raise FileNotFoundError(f'Knowledge granulation is missing {len(missing_inputs)} input files:\n{missing_preview}')
existing_outputs = [job['output_path'] for job in jobs if job['output_path'].exists()]
if existing_outputs and (not cfg.OVERWRITE):
    existing_preview = '\n'.join((str(path) for path in existing_outputs[:20]))
    raise FileExistsError(f'Found {len(existing_outputs)} reclassified outputs already exist and OVERWRITE=False:\n{existing_preview}')
bin_edges_text = ','.join((f'{edge:.12g}' for edge in cfg.BIN_EDGES))
written_count = 0
for job in tqdm(jobs, desc='Knowledge granulation', unit='file', dynamic_ncols=True):
    indicator = job['indicator']
    year = job['year']
    input_path = job['input_path']
    output_path = job['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.stem}.part{output_path.suffix}')
    temporary_path.unlink(missing_ok=True)
    input_valid_count = 0
    output_class_counts = np.zeros(cfg.BIN_COUNT + 1, dtype=np.int64)
    clipped_below_count = 0
    clipped_above_count = 0
    try:
        with rasterio.open(input_path) as source:
            if source.count != cfg.INPUT_BAND_COUNT:
                raise RuntimeError(f'Required: {cfg.INPUT_BAND_COUNT} bands; actual: {source.count}')
            if source.dtypes[0] != cfg.INPUT_DTYPE:
                raise RuntimeError(f'Required input type: {cfg.INPUT_DTYPE}; actual value: {source.dtypes[0]}')
            if source.nodata is None or not np.isclose(source.nodata, cfg.INPUT_NODATA, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE):
                raise RuntimeError(f'Required input NoData={cfg.INPUT_NODATA}; actual value: {source.nodata}')
            source_tags = source.tags()
            expected_period_start = f'{year}-{cfg.PERIOD_START_MONTH:02d}'
            expected_period_end = f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}'
            if source_tags.get('STATISTIC') != cfg.STATISTIC_NAME:
                raise RuntimeError(f"STATISTIC metadata mismatch: {source_tags.get('STATISTIC')}")
            if source_tags.get('PERIOD_START') != expected_period_start:
                raise RuntimeError(f"PERIOD_START metadata mismatch: {source_tags.get('PERIOD_START')}")
            if source_tags.get('PERIOD_END') != expected_period_end:
                raise RuntimeError(f"PERIOD_END metadata mismatch: {source_tags.get('PERIOD_END')}")
            output_profile = source.profile.copy()
            output_profile.pop('nbits', None)
            output_profile.update(driver='GTiff', count=1, dtype=cfg.RECLASS_DTYPE, nodata=cfg.RECLASS_NODATA, tiled=cfg.GTIFF_TILED, blockxsize=cfg.GTIFF_BLOCK_X_SIZE, blockysize=cfg.GTIFF_BLOCK_Y_SIZE, compress=cfg.GTIFF_COMPRESSION, predictor=cfg.INTEGER_PREDICTOR, zlevel=cfg.GTIFF_ZLEVEL, bigtiff=cfg.GTIFF_BIGTIFF)
            with rasterio.open(temporary_path, 'w', **output_profile) as destination:
                for _, window in source.block_windows(1):
                    source_data = source.read(1, window=window, masked=False)
                    source_mask = source.read_masks(1, window=window) > 0
                    finite_mask = np.isfinite(source_data)
                    configured_nodata_mask = source_data == cfg.INPUT_NODATA
                    valid_mask = source_mask & finite_mask & ~configured_nodata_mask
                    output_data = np.full(source_data.shape, cfg.RECLASS_NODATA, dtype=np.uint8)
                    if np.any(valid_mask):
                        valid_values = source_data[valid_mask].astype(np.float64, copy=False)
                        clipped_below_count += int(np.count_nonzero(valid_values < 0.0))
                        clipped_above_count += int(np.count_nonzero(valid_values > cfg.INPUT_SCALE_MAX))
                        normalized_values = valid_values / cfg.INPUT_SCALE_MAX
                        if cfg.CLIP_NORMALIZED_VALUES:
                            normalized_values = np.clip(normalized_values, cfg.NORMALIZED_MIN, cfg.NORMALIZED_MAX)
                        elif np.any((normalized_values < cfg.NORMALIZED_MIN) | (normalized_values > cfg.NORMALIZED_MAX)):
                            raise RuntimeError('Normalized values fall outside [0, 1] while clipping is disabled')
                        class_values = np.floor(normalized_values * cfg.BIN_COUNT).astype(np.int16) + 1
                        class_values = np.clip(class_values, cfg.RECLASS_MIN_VALUE, cfg.RECLASS_MAX_VALUE).astype(np.uint8)
                        output_data[valid_mask] = class_values
                        input_valid_count += int(class_values.size)
                        output_class_counts += np.bincount(class_values, minlength=cfg.BIN_COUNT + 1)[:cfg.BIN_COUNT + 1]
                    destination.write(output_data, 1, window=window)
                destination.set_band_description(1, f'{indicator}_knowledge_granulation_{cfg.BIN_COUNT}_classes')
                destination.update_tags(**source_tags)
                destination.update_tags(PROCESS_STEP='03knowledge_granulation', INDICATOR=indicator, YEAR=str(year), PREPROCESS_INDICATORS=','.join(cfg.PREPROCESS_INDICATORS), PREPROCESS_INDICATOR_COUNT=str(len(cfg.PREPROCESS_INDICATORS)), INPUT_FILE=str(input_path), INPUT_NODATA=str(cfg.INPUT_NODATA), INPUT_SCALE_MAX=str(cfg.INPUT_SCALE_MAX), NORMALIZED_MIN=str(cfg.NORMALIZED_MIN), NORMALIZED_MAX=str(cfg.NORMALIZED_MAX), BIN_COUNT=str(cfg.BIN_COUNT), BIN_EDGES=bin_edges_text, CLASS_MIN=str(cfg.RECLASS_MIN_VALUE), CLASS_MAX=str(cfg.RECLASS_MAX_VALUE), CLASS_INTERVAL_RULE='left-closed right-open; last interval right-closed', OUTPUT_DTYPE=cfg.RECLASS_DTYPE, OUTPUT_NODATA=str(cfg.RECLASS_NODATA), VALID_PIXEL_COUNT=str(input_valid_count), CLIPPED_BELOW_COUNT=str(clipped_below_count), CLIPPED_ABOVE_COUNT=str(clipped_above_count), VALID_AREA_POLICY="retain each indicator's own valid area; common mask not applied", DIRECTION_STATUS=cfg.INDICATOR_CONFIG[indicator]['direction_status'])
        with rasterio.open(temporary_path) as check:
            structural_check = check.count == 1 and check.width == output_profile['width'] and (check.height == output_profile['height']) and (check.dtypes[0] == cfg.RECLASS_DTYPE) and (check.nodata == cfg.RECLASS_NODATA) and (check.crs == output_profile['crs']) and check.transform.almost_equals(output_profile['transform']) and (check.tags().get('PROCESS_STEP') == '03knowledge_granulation') and (check.tags().get('INDICATOR') == indicator) and (check.tags().get('YEAR') == str(year)) and (check.tags().get('BIN_COUNT') == str(cfg.BIN_COUNT)) and (check.tags().get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (check.tags().get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS)))
            if not structural_check:
                raise RuntimeError('Temporary reclassified raster failed structure or metadata checks')
            check_valid_count = 0
            invalid_class_count = 0
            for _, window in check.block_windows(1):
                check_data = check.read(1, window=window, masked=False)
                check_valid = check_data != cfg.RECLASS_NODATA
                check_valid_values = check_data[check_valid]
                check_valid_count += int(check_valid_values.size)
                invalid_class_count += int(np.count_nonzero((check_valid_values < cfg.RECLASS_MIN_VALUE) | (check_valid_values > cfg.RECLASS_MAX_VALUE)))
            if check_valid_count != input_valid_count:
                raise RuntimeError(f'Temporary reclassified valid-pixel count differs from input; input {input_valid_count}; output {check_valid_count}')
            if invalid_class_count:
                raise RuntimeError(f'Temporary reclassified raster contains {invalid_class_count} invalid class values')
            if int(output_class_counts[1:].sum()) != input_valid_count:
                raise RuntimeError('In-memory class counts do not match valid input pixels')
        os.replace(temporary_path, output_path)
        written_count += 1
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f'Knowledge granulation failed: {indicator} {year}\nInput: {input_path}\nOutput: {output_path}\n{error}') from error
print(f'Knowledge granulation completed; wrote {written_count} reclassified rasters.')
print(f'Preprocessing indicator count: {len(cfg.PREPROCESS_INDICATORS)}')
print(f'Output directory: {cfg.GRANULATION_OUTPUT_DIR}')
