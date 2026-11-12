"""Calculate DSI as the weighted sum of continuous normalized indicators."""
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
from tqdm import tqdm
import config as cfg
if tuple(cfg.DI_OUTPUT_DIRS) != cfg.DI_WEIGHT_MODES:
    raise RuntimeError('DI_OUTPUT_DIRS does not match DI_WEIGHT_MODES')
if not cfg.GK_ER_W_XLSX.is_file():
    raise FileNotFoundError(f'Missing step 6 Gk/Er/W workbook: {cfg.GK_ER_W_XLSX}')
if not cfg.SELECTED_INDICATORS:
    raise ValueError('SELECTED_INDICATORS must not be empty')
calculation_indicators = cfg.SELECTED_INDICATORS_IN_CONFIG_ORDER
expected_weight_columns = [*cfg.YEARS, cfg.STUDY_PERIOD_SHEET_NAME]
if cfg.INPUT_SCALE_MAX <= 0:
    raise ValueError('INPUT_SCALE_MAX must be positive')
if cfg.DI_NODATA >= cfg.DI_VALID_MIN and cfg.DI_NODATA <= cfg.DI_VALID_MAX:
    raise ValueError('DI_NODATA must lie outside the valid DSI range')
weight_table = pd.read_excel(cfg.GK_ER_W_XLSX, sheet_name='W', index_col=0, engine=cfg.EXCEL_ENGINE)
if str(weight_table.index.name) != cfg.EXCEL_INDEX_LABEL:
    raise RuntimeError(f'Worksheet W first-column title must be {cfg.EXCEL_INDEX_LABEL}：{weight_table.index.name}')
actual_weight_indicators = [str(value) for value in weight_table.index]
expected_weight_indicators = list(calculation_indicators)
if actual_weight_indicators != expected_weight_indicators:
    raise RuntimeError(f'Worksheet W indicators or order differ from the current indicator group.\nRequired: {expected_weight_indicators}\nActual: {actual_weight_indicators}')
actual_weight_columns = []
for value in weight_table.columns:
    if isinstance(value, (int, np.integer)):
        actual_weight_columns.append(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        actual_weight_columns.append(int(value))
    else:
        normalized_column = str(value).strip()
        if normalized_column == cfg.STUDY_PERIOD_SHEET_NAME:
            actual_weight_columns.append(normalized_column)
        else:
            try:
                actual_weight_columns.append(int(normalized_column))
            except ValueError as error:
                raise RuntimeError(f'Worksheet W has invalid period columns: {value}') from error
if actual_weight_columns != expected_weight_columns:
    raise RuntimeError(f'Worksheet W year and study-period columns differ from configuration.\nRequired: {expected_weight_columns}\nActual: {actual_weight_columns}')
weight_table.columns = actual_weight_columns
weight_values = weight_table.to_numpy(dtype=np.float64)
if not bool(np.isfinite(weight_values).all()):
    raise RuntimeError('Worksheet W contains NaN or Inf')
if bool(((weight_values < -cfg.VALUE_RANGE_TOLERANCE) | (weight_values > 1.0 + cfg.VALUE_RANGE_TOLERANCE)).any()):
    raise RuntimeError('Worksheet W contains weights outside [0, 1]')
period_weight_sums = weight_values.sum(axis=0, dtype=np.float64)
if not bool(np.isclose(period_weight_sums, 1.0, rtol=0.0, atol=cfg.WEIGHT_SUM_TOLERANCE).all()):
    raise RuntimeError(f'Worksheet W contains period-weight sums other than one: {period_weight_sums.tolist()}')
jobs = []
for weight_mode in cfg.DI_WEIGHT_MODES:
    if weight_mode == 'annual_weights':
        weight_column_by_year = {year: year for year in cfg.YEARS}
    elif weight_mode == 'study_period_weights':
        weight_column_by_year = {year: cfg.STUDY_PERIOD_SHEET_NAME for year in cfg.YEARS}
    else:
        raise ValueError(f'Unsupported DSI weighting mode: {weight_mode}')
    for year in cfg.YEARS:
        source_paths = {indicator: cfg.INDICATOR_CONFIG[indicator]['input_dir'] / cfg.INPUT_FILE_PATTERN.format(indicator=indicator, year=year) for indicator in calculation_indicators}
        jobs.append({'weight_mode': weight_mode, 'weight_column': weight_column_by_year[year], 'year': year, 'source_paths': source_paths, 'mask_path': cfg.VALID_MASK_OUTPUT_DIR / cfg.MASK_FILE_PATTERN.format(year=year), 'output_path': cfg.DI_OUTPUT_DIRS[weight_mode] / cfg.DI_FILE_PATTERN.format(year=year)})
missing_inputs = []
for job in jobs:
    if not job['mask_path'].is_file():
        missing_inputs.append(job['mask_path'])
    for source_path in job['source_paths'].values():
        if not source_path.is_file():
            missing_inputs.append(source_path)
if missing_inputs:
    missing_preview = '\n'.join((str(path) for path in missing_inputs[:20]))
    raise FileNotFoundError(f'Step 7 is missing {len(missing_inputs)} input files:\n{missing_preview}')
existing_outputs = [job['output_path'] for job in jobs if job['output_path'].exists()]
if existing_outputs and (not cfg.OVERWRITE):
    existing_preview = '\n'.join((str(path) for path in existing_outputs[:20]))
    raise FileExistsError(f'Found {len(existing_outputs)} DSI outputs already exist and OVERWRITE=False:\n{existing_preview}')
written_di_count_by_mode = {weight_mode: 0 for weight_mode in cfg.DI_WEIGHT_MODES}
for job in tqdm(jobs, desc='Calculate both DSI weighting modes', unit='file', dynamic_ncols=True):
    weight_mode = job['weight_mode']
    weight_column = job['weight_column']
    year = job['year']
    source_paths = job['source_paths']
    mask_path = job['mask_path']
    output_path = job['output_path']
    applied_weights = weight_table.loc[list(calculation_indicators), weight_column].to_numpy(dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.stem}.part{output_path.suffix}')
    temporary_path.unlink(missing_ok=True)
    common_valid_count = 0
    di_value_count = 0
    di_value_sum = 0.0
    di_value_square_sum = 0.0
    di_value_min = np.inf
    di_value_max = -np.inf
    clipped_low_counts = {indicator: 0 for indicator in calculation_indicators}
    clipped_high_counts = {indicator: 0 for indicator in calculation_indicators}
    try:
        with ExitStack() as stack:
            mask_source = stack.enter_context(rasterio.open(mask_path))
            sources = {indicator: stack.enter_context(rasterio.open(path)) for indicator, path in source_paths.items()}
            reference_indicator = calculation_indicators[0]
            reference_source = sources[reference_indicator]
            reference_transform = np.asarray(tuple(reference_source.transform), dtype=np.float64)
            mask_tags = mask_source.tags()
            mask_image_structure = mask_source.tags(1, ns='IMAGE_STRUCTURE')
            mask_transform_matches = np.allclose(np.asarray(tuple(mask_source.transform), dtype=np.float64), reference_transform, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
            mask_ok = mask_source.count == 1 and mask_source.width == reference_source.width and (mask_source.height == reference_source.height) and (mask_source.crs == reference_source.crs) and mask_transform_matches and (mask_source.dtypes[0] == cfg.MASK_DTYPE) and (mask_source.nodata == cfg.MASK_NODATA) and (mask_image_structure.get('NBITS') == str(cfg.MASK_NBITS)) and (mask_tags.get('PROCESS_STEP') == '04valid_mask') and (mask_tags.get('YEAR') == str(year)) and (mask_tags.get('STATISTIC') == cfg.STATISTIC_NAME) and (mask_tags.get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (mask_tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS))) and (mask_tags.get('VALIDITY_SOURCE') == 'knowledge_granulation_classes')
            if not mask_ok:
                raise RuntimeError(f'Annual common-valid mask failed structure or metadata checks: {mask_path}')
            for indicator, source in sources.items():
                source_transform_matches = np.allclose(np.asarray(tuple(source.transform), dtype=np.float64), reference_transform, rtol=0.0, atol=cfg.GRID_ABSOLUTE_TOLERANCE)
                source_tags = source.tags()
                source_ok = source.count == cfg.INPUT_BAND_COUNT and source.width == reference_source.width and (source.height == reference_source.height) and (source.crs == reference_source.crs) and source_transform_matches and (source.dtypes[0] == cfg.INPUT_DTYPE) and (source.nodata is not None) and np.isclose(source.nodata, cfg.INPUT_NODATA, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE) and (source_tags.get('STATISTIC') == cfg.STATISTIC_NAME) and (source_tags.get('PERIOD_KIND') == cfg.PERIOD_KIND)
                if not source_ok:
                    raise RuntimeError(f'Continuous indicator raster failed structure, grid, or metadata checks: {indicator} {year} {source_paths[indicator]}')
            output_profile = reference_source.profile.copy()
            output_profile.update(driver='GTiff', count=1, dtype=cfg.DI_DTYPE, nodata=cfg.DI_NODATA, tiled=cfg.GTIFF_TILED, blockxsize=cfg.GTIFF_BLOCK_X_SIZE, blockysize=cfg.GTIFF_BLOCK_Y_SIZE, compress=cfg.GTIFF_COMPRESSION, zlevel=cfg.GTIFF_ZLEVEL, predictor=cfg.FLOAT_PREDICTOR, bigtiff=cfg.GTIFF_BIGTIFF)
            with rasterio.open(temporary_path, 'w', **output_profile) as destination:
                for _, window in mask_source.block_windows(1):
                    mask_data = mask_source.read(1, window=window, masked=False)
                    legal_mask_codes = (mask_data == cfg.MASK_INVALID_VALUE) | (mask_data == cfg.MASK_VALID_VALUE)
                    if not bool(legal_mask_codes.all()):
                        illegal_count = int(np.count_nonzero(~legal_mask_codes))
                        raise RuntimeError(f'{year} annual mask tile contains {illegal_count} codes other than 0 or 1')
                    common_valid = mask_data == cfg.MASK_VALID_VALUE
                    block_valid_count = int(common_valid.sum())
                    common_valid_count += block_valid_count
                    output_data = np.full(mask_data.shape, cfg.DI_NODATA, dtype=np.float32)
                    if block_valid_count:
                        di_valid_values = np.zeros(block_valid_count, dtype=np.float64)
                        for indicator_index, indicator in enumerate(calculation_indicators):
                            source = sources[indicator]
                            source_data = source.read(1, window=window, masked=False)
                            source_mask = source.read_masks(1, window=window) > 0
                            source_valid = source_mask & np.isfinite(source_data) & (source_data != cfg.INPUT_NODATA)
                            invalid_inside_common = common_valid & ~source_valid
                            invalid_inside_count = int(invalid_inside_common.sum())
                            if invalid_inside_count:
                                raise RuntimeError(f'{indicator} {year} contains within the common mask {invalid_inside_count} invalid values')
                            raw_valid_values = source_data[common_valid].astype(np.float64)
                            clipped_low_counts[indicator] += int(np.count_nonzero(raw_valid_values < 0.0))
                            clipped_high_counts[indicator] += int(np.count_nonzero(raw_valid_values > cfg.INPUT_SCALE_MAX))
                            normalized_values = raw_valid_values / cfg.INPUT_SCALE_MAX
                            if cfg.CLIP_NORMALIZED_VALUES:
                                normalized_values = np.clip(normalized_values, cfg.NORMALIZED_MIN, cfg.NORMALIZED_MAX)
                            di_valid_values += applied_weights[indicator_index] * normalized_values
                        di_valid_values = np.clip(di_valid_values, cfg.DI_VALID_MIN, cfg.DI_VALID_MAX)
                        output_data[common_valid] = di_valid_values.astype(np.float32)
                        di_value_count += block_valid_count
                        di_value_sum += float(di_valid_values.sum(dtype=np.float64))
                        di_value_square_sum += float(np.square(di_valid_values).sum(dtype=np.float64))
                        di_value_min = min(di_value_min, float(di_valid_values.min()))
                        di_value_max = max(di_value_max, float(di_valid_values.max()))
                    destination.write(output_data, 1, window=window)
                if common_valid_count <= 0:
                    raise RuntimeError(f'{year} annual common-valid pixel count is zero')
                if di_value_count != common_valid_count:
                    raise RuntimeError(f'{year} annual valid DSI count differs from the common mask')
                weight_text = ';'.join((f'{indicator}={applied_weights[index]:.17g}' for index, indicator in enumerate(calculation_indicators)))
                clipped_low_text = ';'.join((f'{indicator}={clipped_low_counts[indicator]}' for indicator in calculation_indicators))
                clipped_high_text = ';'.join((f'{indicator}={clipped_high_counts[indicator]}' for indicator in calculation_indicators))
                destination.set_band_description(1, f'DI_{year}')
                destination.update_tags(PROCESS_STEP='07calculate_DI', YEAR=str(year), STATISTIC=cfg.STATISTIC_NAME, STATISTIC_DIR_NAME=cfg.STATISTIC_DIR_NAME, PERIOD_KIND=cfg.PERIOD_KIND, PERIOD_START=f'{year}-{cfg.PERIOD_START_MONTH:02d}', PERIOD_END=f'{year + cfg.PERIOD_END_YEAR_OFFSET}-{cfg.PERIOD_END_MONTH:02d}', INDICATOR_GROUP_NAME=cfg.INDICATOR_GROUP_NAME, SELECTED_INDICATORS=','.join(calculation_indicators), SELECTED_INDICATOR_COUNT=str(len(calculation_indicators)), PREPROCESS_INDICATORS=','.join(cfg.PREPROCESS_INDICATORS), PREPROCESS_INDICATOR_COUNT=str(len(cfg.PREPROCESS_INDICATORS)), WEIGHT_SOURCE=str(cfg.GK_ER_W_XLSX), WEIGHT_SHEET='W', WEIGHT_MODE=weight_mode, WEIGHT_COLUMN=str(weight_column), STUDY_PERIOD_WEIGHT_COLUMN=cfg.STUDY_PERIOD_SHEET_NAME, APPLIED_WEIGHTS=weight_text, WEIGHT_SUM=f'{float(applied_weights.sum()):.17g}', DI_FORMULA='DI=sum(W_k*clip(x_k/10000,0,1))', INPUT_SCALE_MAX=str(cfg.INPUT_SCALE_MAX), DIRECTION_STATUS='all indicators already unified; higher means more desertification', COMMON_VALID_PIXEL_COUNT=str(common_valid_count), DI_VALID_PIXEL_COUNT=str(di_value_count), DI_MIN=f'{di_value_min:.17g}', DI_MAX=f'{di_value_max:.17g}', DI_MEAN=f'{di_value_sum / di_value_count:.17g}', CLIPPED_LOW_COUNTS=clipped_low_text, CLIPPED_HIGH_COUNTS=clipped_high_text, OUTPUT_DTYPE=cfg.DI_DTYPE, OUTPUT_NODATA=str(cfg.DI_NODATA))
        with rasterio.open(temporary_path) as check:
            check_tags = check.tags()
            structural_check = check.count == 1 and check.width == output_profile['width'] and (check.height == output_profile['height']) and (check.dtypes[0] == cfg.DI_DTYPE) and (check.nodata == cfg.DI_NODATA) and (check.crs == output_profile['crs']) and check.transform.almost_equals(output_profile['transform']) and (check_tags.get('PROCESS_STEP') == '07calculate_DI') and (check_tags.get('YEAR') == str(year)) and (check_tags.get('STATISTIC') == cfg.STATISTIC_NAME) and (check_tags.get('INDICATOR_GROUP_NAME') == cfg.INDICATOR_GROUP_NAME) and (check_tags.get('SELECTED_INDICATORS') == ','.join(calculation_indicators)) and (check_tags.get('SELECTED_INDICATOR_COUNT') == str(len(calculation_indicators))) and (check_tags.get('PREPROCESS_INDICATORS') == ','.join(cfg.PREPROCESS_INDICATORS)) and (check_tags.get('PREPROCESS_INDICATOR_COUNT') == str(len(cfg.PREPROCESS_INDICATORS))) and (check_tags.get('WEIGHT_MODE') == weight_mode) and (check_tags.get('WEIGHT_COLUMN') == str(weight_column))
            if not structural_check:
                raise RuntimeError(f'{year} annual temporary DSI failed structure or metadata checks')
            check_valid_count = 0
            check_nodata_count = 0
            check_nan_or_inf_count = 0
            check_below_count = 0
            check_above_count = 0
            check_min = np.inf
            check_max = -np.inf
            for _, window in check.block_windows(1):
                check_data = check.read(1, window=window, masked=False)
                nodata_mask = check_data == cfg.DI_NODATA
                valid_mask = ~nodata_mask
                valid_values = check_data[valid_mask]
                check_nodata_count += int(nodata_mask.sum())
                check_valid_count += int(valid_mask.sum())
                if valid_values.size:
                    finite_mask = np.isfinite(valid_values)
                    check_nan_or_inf_count += int(np.count_nonzero(~finite_mask))
                    finite_values = valid_values[finite_mask]
                    if finite_values.size:
                        check_below_count += int(np.count_nonzero(finite_values < cfg.DI_VALID_MIN - cfg.VALUE_RANGE_TOLERANCE))
                        check_above_count += int(np.count_nonzero(finite_values > cfg.DI_VALID_MAX + cfg.VALUE_RANGE_TOLERANCE))
                        check_min = min(check_min, float(finite_values.min()))
                        check_max = max(check_max, float(finite_values.max()))
            total_pixel_count = check.width * check.height
            if check_valid_count != common_valid_count:
                raise RuntimeError(f'{year} annual temporary DSI has an incorrect valid-pixel count; required {common_valid_count}; actual: {check_valid_count}')
            if check_nodata_count != total_pixel_count - common_valid_count:
                raise RuntimeError(f'{year} annual temporary DSI has an incorrect NoData pixel count')
            if check_nan_or_inf_count:
                raise RuntimeError(f'{year} annual temporary DSI contains {check_nan_or_inf_count} NaN/Inf values')
            if check_below_count or check_above_count:
                raise RuntimeError(f'{year} annual temporary DSI is outside [0, 1]; low values {check_below_count}; high values: {check_above_count}')
            if not np.isclose(check_min, di_value_min, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE) or not np.isclose(check_max, di_value_max, rtol=0.0, atol=cfg.VALUE_RANGE_TOLERANCE):
                raise RuntimeError(f'{year} annual temporary DSI minimum or maximum verification mismatch')
        os.replace(temporary_path, output_path)
        written_di_count_by_mode[weight_mode] += 1
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f'DSI calculation failed: {weight_mode} {year}\nOutput: {output_path}\n{error}') from error
print(f'DSI calculation completed; wrote {sum(written_di_count_by_mode.values())} rasters.')
for weight_mode in cfg.DI_WEIGHT_MODES:
    print(f'{weight_mode}：{written_di_count_by_mode[weight_mode]} rasters; directory: {cfg.DI_OUTPUT_DIRS[weight_mode]}')
print(f'DSI filename pattern: {cfg.DI_FILE_PATTERN}')
