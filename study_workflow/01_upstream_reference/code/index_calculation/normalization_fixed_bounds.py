"""Normalize indicator rasters using fixed bounds."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, UPSTREAM_DATA_ROOT, raster_source
import glob
import math
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from openpyxl import load_workbook
from osgeo import gdal
from tqdm import tqdm
in_root = VRT_ROOT / 'rs_indices'
out_root = UPSTREAM_DATA_ROOT / 'rs_normalized'
bounds_xlsx = UPSTREAM_DATA_ROOT / 'statistics/index_fixed_bounds.xlsx'
mask_path = raster_source(VRT_ROOT / 'auxiliary_sources/study_mask_native', 'arid_region_mask')
lst_source_root = VRT_ROOT / 'lst'
test_mode = False
test_year, test_month = (2001, 1)
start_year, end_year = (2001, 2022)
start_month, end_month = (1, 12)
overwrite = False
max_processes = 3
gdal_threads_per_process = 2
gdal_cache_mb_per_process = 512
indices = {'Albedo': True, 'SMI': True, 'TGSI': True, 'LST': True, 'DFI': False, 'MSAVI': False, 'Wet': False}
in_exts = ['.tif', '.vrt']
out_nodata = 32767
block_size = 2048
error_log = os.path.join(out_root, 'error_log.txt')

def process_file(task):
    gdal.UseExceptions()
    gdal.SetConfigOption('VRT_SHARED_SOURCE', '0')
    gdal.SetConfigOption('GDAL_CACHEMAX', str(gdal_cache_mb_per_process))
    gdal.SetCacheMax(gdal_cache_mb_per_process * 1024 * 1024)
    index_name = task['index_name']
    img_path = task['img_path']
    out_path = task['out_path']
    positive = task['positive']
    lower_bound = task['lower_bound']
    upper_bound = task['upper_bound']
    valid_windows = task['valid_windows']
    mask_xsize = task['mask_xsize']
    mask_ysize = task['mask_ysize']
    mask_geotransform = task['mask_geotransform']
    mask_projection = task['mask_projection']
    if os.path.exists(out_path) and (not overwrite):
        return {'status': 'skipped', 'index_name': index_name, 'img_path': img_path, 'out_path': out_path}
    tmp_path = out_path + '.tmp'
    repaired_vrt_path = None
    ds = None
    band = None
    out_ds = None
    out_band = None
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        ext = os.path.splitext(img_path)[1].lower()
        open_path = img_path
        if ext == '.vrt':
            with open(img_path, 'r', encoding='utf-8') as vrt_file:
                vrt_text = vrt_file.read()
            old_source_prefix = 'relativeToVRT="1">..\\..\\..\\03orig\\'
            if old_source_prefix in vrt_text:
                if not os.path.isdir(lst_source_root):
                    raise FileNotFoundError(f'LST source-data directory not found: {lst_source_root}')
                new_source_prefix = f'relativeToVRT="0">{lst_source_root}\\'
                vrt_text = vrt_text.replace(old_source_prefix, new_source_prefix)
                repaired_vrt_path = out_path + '.source.vrt'
                with open(repaired_vrt_path, 'w', encoding='utf-8') as vrt_file:
                    vrt_file.write(vrt_text)
                open_path = repaired_vrt_path
        open_options = [f'NUM_THREADS={gdal_threads_per_process}'] if ext == '.tif' else []
        ds = gdal.OpenEx(open_path, gdal.OF_RASTER, open_options=open_options)
        if ds is None:
            raise RuntimeError('GDAL cannot open input raster')
        if ds.RasterCount != 1:
            raise ValueError(f'Only single-band rasters are supported; actual band count {ds.RasterCount}')
        if (ds.RasterXSize, ds.RasterYSize) != (mask_xsize, mask_ysize):
            raise ValueError(f'Input dimensions differ from the study mask: {ds.RasterXSize}×{ds.RasterYSize} != {mask_xsize}×{mask_ysize}')
        if ds.GetGeoTransform() != mask_geotransform:
            raise ValueError('Input affine transform differs from the study mask')
        if ds.GetProjection() != mask_projection:
            raise ValueError('Input projection differs from the study mask')
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        nodata_has_value = nodata is not None and (not (isinstance(nodata, float) and math.isnan(nodata)))
        creation_options = ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES', 'SPARSE_OK=TRUE', f'NUM_THREADS={gdal_threads_per_process}', 'BIGTIFF=IF_SAFER']
        out_ds = gdal.GetDriverByName('GTiff').Create(tmp_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Int16, creation_options)
        if out_ds is None:
            raise RuntimeError('GDAL cannot create temporary output raster')
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(out_nodata)
        out_band.Fill(out_nodata)
        valid_pixels = 0
        output_min = out_nodata
        output_max = -1
        clipped_to_zero = 0
        clipped_to_one = 0
        for xoff, yoff in valid_windows:
            xcount = min(block_size, ds.RasterXSize - xoff)
            ycount = min(block_size, ds.RasterYSize - yoff)
            arr = band.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float32, copy=False)
            valid = np.isfinite(arr)
            if nodata_has_value:
                valid &= arr != nodata
            if not valid.any():
                continue
            values = arr[valid]
            if positive:
                clipped_to_zero += int(np.count_nonzero(values <= lower_bound))
                clipped_to_one += int(np.count_nonzero(values >= upper_bound))
            else:
                clipped_to_zero += int(np.count_nonzero(values >= upper_bound))
                clipped_to_one += int(np.count_nonzero(values <= lower_bound))
            norm = (arr - lower_bound) / (upper_bound - lower_bound)
            np.clip(norm, 0.0, 1.0, out=norm)
            if not positive:
                norm = 1.0 - norm
            out_arr = np.full(arr.shape, out_nodata, dtype=np.int16)
            out_arr[valid] = np.rint(norm[valid] * 10000.0).astype(np.int16)
            out_band.WriteArray(out_arr, xoff, yoff)
            block_output = out_arr[valid]
            valid_pixels += int(block_output.size)
            output_min = min(output_min, int(block_output.min()))
            output_max = max(output_max, int(block_output.max()))
        if valid_pixels == 0:
            raise ValueError('Input raster has no valid pixels')
        if output_min < 0 or output_max > 10000:
            raise ValueError(f'Invalid output range: {output_min}—{output_max}')
        out_band.FlushCache()
        out_band = None
        out_ds = None
        band = None
        ds = None
        if repaired_vrt_path is not None and os.path.exists(repaired_vrt_path):
            os.remove(repaired_vrt_path)
        for replace_attempt in range(10):
            try:
                os.replace(tmp_path, out_path)
                break
            except PermissionError:
                if replace_attempt == 9:
                    raise
                time.sleep(1)
        return {'status': 'completed', 'index_name': index_name, 'img_path': img_path, 'out_path': out_path, 'valid_pixels': valid_pixels, 'output_min': output_min, 'output_max': output_max, 'clipped_to_zero': clipped_to_zero, 'clipped_to_one': clipped_to_one}
    except Exception:
        out_band = None
        out_ds = None
        band = None
        ds = None
        for temporary_path in (tmp_path, repaired_vrt_path):
            if temporary_path is not None and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
        return {'status': 'failed', 'index_name': index_name, 'img_path': img_path, 'out_path': out_path, 'traceback': traceback.format_exc()}
if __name__ == '__main__':
    gdal.UseExceptions()
    gdal.SetConfigOption('VRT_SHARED_SOURCE', '0')
    if not os.path.isfile(bounds_xlsx):
        raise FileNotFoundError(f'Fixed-bounds workbook not found: {bounds_xlsx}')
    bounds_wb = load_workbook(bounds_xlsx, read_only=True, data_only=True)
    if 'fixed_bounds' not in bounds_wb.sheetnames:
        bounds_wb.close()
        raise ValueError(f'Fixed-bounds workbook lacks the fixed_bounds sheet ({bounds_xlsx}）')
    bounds_ws = bounds_wb['fixed_bounds']
    header_values = [cell.value for cell in next(bounds_ws.iter_rows(min_row=1, max_row=1))]
    header_index = {value: position for position, value in enumerate(header_values)}
    required_headers = ['Indicator', 'Fixed lower bound (minimum P1)', 'Fixed upper bound (maximum P99)']
    missing_headers = [name for name in required_headers if name not in header_index]
    if missing_headers:
        bounds_wb.close()
        raise ValueError(f"Fixed-bounds workbook lacks fields: {', '.join(missing_headers)}")
    fixed_bounds = {}
    for row_number, row in enumerate(bounds_ws.iter_rows(min_row=2, values_only=True), start=2):
        index_name = row[header_index['Indicator']]
        if index_name in (None, ''):
            continue
        index_name = str(index_name).strip()
        if index_name in fixed_bounds:
            bounds_wb.close()
            raise ValueError(f'Fixed-bounds table row {row_number} row contains duplicate indicator: {index_name}')
        try:
            lower_bound = float(row[header_index['Fixed lower bound (minimum P1)']])
            upper_bound = float(row[header_index['Fixed upper bound (maximum P99)']])
        except (TypeError, ValueError) as exc:
            bounds_wb.close()
            raise ValueError(f'Fixed-bounds table row {row_number} row bounds are not numeric') from exc
        if not (math.isfinite(lower_bound) and math.isfinite(upper_bound)):
            bounds_wb.close()
            raise ValueError(f'Fixed-bounds table row {row_number} row bounds contain NaN or infinity')
        if upper_bound <= lower_bound:
            bounds_wb.close()
            raise ValueError(f'{index_name} fixed upper bound ({upper_bound}) does not exceed lower bound ({lower_bound})')
        fixed_bounds[index_name] = (lower_bound, upper_bound)
    bounds_wb.close()
    missing_indices = [name for name in indices if name not in fixed_bounds]
    extra_indices = [name for name in fixed_bounds if name not in indices]
    if missing_indices:
        raise ValueError(f"Fixed-bounds workbook lacks indicators: {', '.join(missing_indices)}")
    if extra_indices:
        print('Indicators in the fixed-bounds table omitted from processing: ', ', '.join(extra_indices))
    if os.path.isdir(out_root):
        residual_patterns = ['*.tif.tmp', '*.tif.source.vrt']
        for residual_pattern in residual_patterns:
            for temporary_path in glob.glob(os.path.join(out_root, '**', residual_pattern), recursive=True):
                print('Remove residual temporary file: ', temporary_path)
                os.remove(temporary_path)
    tasks = []
    missing_test_inputs = []
    index_order = {name: position for position, name in enumerate(indices)}
    for index_name, positive in indices.items():
        in_dir = str(lst_source_root if index_name == 'LST' else UPSTREAM_DATA_ROOT / 'rs_indices' / index_name if os.environ.get('REPRO_REBUILT_INDICES') == '1' else in_root / index_name)
        if not os.path.isdir(in_dir):
            if test_mode:
                missing_test_inputs.append(in_dir)
            else:
                raise FileNotFoundError('Missing indicator directory: ' + in_dir)
            continue
        for year_name in sorted(os.listdir(in_dir)):
            year_dir = os.path.join(in_dir, year_name)
            if not os.path.isdir(year_dir) or not year_name.isdigit():
                continue
            year_dir_value = int(year_name)
            if test_mode:
                if year_dir_value != test_year:
                    continue
            elif not start_year <= year_dir_value <= end_year:
                continue
            found = {}
            for ext in in_exts:
                for img_path in glob.glob(os.path.join(year_dir, '*' + ext)):
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    if stem not in found:
                        found[stem] = img_path
            for stem in sorted(found):
                parts = stem.split('_')
                if len(parts) < 3 or not parts[-2].isdigit() or (not parts[-1].isdigit()):
                    print('Cannot parse year/month from filename; skipped: ', found[stem])
                    continue
                year_value, month_value = (int(parts[-2]), int(parts[-1]))
                if test_mode:
                    if year_value != test_year or month_value != test_month:
                        continue
                elif not (start_year <= year_value <= end_year and start_month <= month_value <= end_month):
                    continue
                out_path = os.path.join(out_root, index_name, str(year_value), stem + '.tif')
                tasks.append({'index_name': index_name, 'img_path': found[stem], 'out_path': out_path, 'positive': positive, 'lower_bound': fixed_bounds[index_name][0], 'upper_bound': fixed_bounds[index_name][1], 'year': year_value, 'month': month_value})
        if test_mode and (not any((task['index_name'] == index_name for task in tasks))):
            missing_test_inputs.append(os.path.join(in_dir, str(test_year), f'{index_name}_{test_year:04d}_{test_month:02d}.tif/.vrt'))
    if missing_test_inputs:
        raise FileNotFoundError('Test mode requires inputs for all seven indicators; missing:\n  ' + '\n  '.join(missing_test_inputs))
    if test_mode and len(tasks) != len(indices):
        raise RuntimeError(f'Test mode should collect {len(indices)} tasks; actual {len(tasks)} items')
    tasks.sort(key=lambda task: (task['year'], task['month'], index_order[task['index_name']]))
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f'Study mask not found: {mask_path}')
    mask_ds = gdal.OpenEx(mask_path, gdal.OF_RASTER, open_options=[f'NUM_THREADS={gdal_threads_per_process}'])
    if mask_ds is None:
        raise RuntimeError(f'GDAL cannot open study mask: {mask_path}')
    if mask_ds.RasterCount != 1:
        raise ValueError(f'Study mask must be single-band; actual {mask_ds.RasterCount} bands')
    mask_band = mask_ds.GetRasterBand(1)
    mask_xsize, mask_ysize = (mask_ds.RasterXSize, mask_ds.RasterYSize)
    mask_geotransform = mask_ds.GetGeoTransform()
    mask_projection = mask_ds.GetProjection()
    mask_nodata = mask_band.GetNoDataValue()
    mask_nodata_has_value = mask_nodata is not None and (not (isinstance(mask_nodata, float) and math.isnan(mask_nodata)))
    valid_windows = []
    all_window_count = 0
    for yoff in range(0, mask_ysize, block_size):
        ycount = min(block_size, mask_ysize - yoff)
        for xoff in range(0, mask_xsize, block_size):
            xcount = min(block_size, mask_xsize - xoff)
            all_window_count += 1
            mask_arr = mask_band.ReadAsArray(xoff, yoff, xcount, ycount)
            mask_valid = np.isfinite(mask_arr)
            if mask_nodata_has_value:
                mask_valid &= mask_arr != mask_nodata
            mask_valid &= mask_arr != 0
            if mask_valid.any():
                valid_windows.append((xoff, yoff))
    mask_band = None
    mask_ds = None
    if not valid_windows:
        raise ValueError('Study mask has no valid windows')
    for task in tasks:
        task['valid_windows'] = valid_windows
        task['mask_xsize'] = mask_xsize
        task['mask_ysize'] = mask_ysize
        task['mask_geotransform'] = mask_geotransform
        task['mask_projection'] = mask_projection
    print(f'Valid study-mask windows: {len(valid_windows)}/{all_window_count}; skipped: {all_window_count - len(valid_windows)} entirely empty windows')
    print('\nFixed normalization bounds: ')
    for index_name in indices:
        lower_bound, upper_bound = fixed_bounds[index_name]
        direction = 'positive' if indices[index_name] else 'negative'
        print(f'  {index_name:<7} lower bound={lower_bound:.15g}  upper bound={upper_bound:.15g}  {direction}')
    if test_mode:
        print(f'\nMode: test ({test_year:04d}-{test_month:02d}）')
    else:
        print(f'\nMode: full ({start_year}-{end_year} years; months per year {start_month}-{end_month} months)')
    if overwrite:
        pending_tasks = tasks
        skipped = 0
    else:
        pending_tasks = [task for task in tasks if not os.path.exists(task['out_path'])]
        skipped = len(tasks) - len(pending_tasks)
    process_count = min(max_processes, len(pending_tasks)) if pending_tasks else 0
    print(f'Task count: {len(tasks)}; existing outputs skipped: {skipped}; tasks to process: {len(pending_tasks)}')
    print(f'Overwrite existing outputs: {overwrite}')
    print(f'Parallel settings: {process_count} file processes; threads per process {gdal_threads_per_process} GDAL threads; cache per process {gdal_cache_mb_per_process} MB cache')
    completed = 0
    failed = []
    pbar = tqdm(total=len(tasks), initial=skipped, desc='Overall progress', unit='file')
    if pending_tasks:
        spawn_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=process_count, mp_context=spawn_context) as executor:
            future_tasks = {executor.submit(process_file, task): task for task in pending_tasks}
            try:
                for future in as_completed(future_tasks):
                    task = future_tasks[future]
                    try:
                        result = future.result()
                    except Exception:
                        result = {'status': 'failed', 'index_name': task['index_name'], 'img_path': task['img_path'], 'out_path': task['out_path'], 'traceback': traceback.format_exc()}
                    if result['status'] == 'completed':
                        completed += 1
                    elif result['status'] == 'skipped':
                        skipped += 1
                    else:
                        failed.append(result)
                        tqdm.write(f"Processing failed: {result['img_path']}(see error log)")
                        os.makedirs(out_root, exist_ok=True)
                        with open(error_log, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {result['img_path']}\n")
                            log_file.write(result['traceback'])
                            log_file.write('-' * 80 + '\n')
                    pbar.update(1)
            except KeyboardInterrupt:
                for future in future_tasks:
                    future.cancel()
                tqdm.write('Interrupted: pending tasks cancelled; waiting for running tasks to clean up')
                raise
    pbar.close()
    print(f'\nProcessing complete; succeeded: {completed}; skipped: {skipped}; failed: {len(failed)} items.')
    if failed:
        print(f'Error log: {error_log}')
        for failed_result in failed:
            print('  ', failed_result['img_path'])
        raise RuntimeError(f'There are {len(failed)} failed files')
    print('Output directory: ', out_root)
