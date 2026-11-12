"""Resample normalized indices to the 0.1-degree INT16 grid."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, raster_source, raster_inputs, project_file
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from osgeo import gdal
from tqdm import tqdm
SOURCE_DIR = project_file(os.environ.get('REPRO_NORMALIZED_ROOT', str(VRT_ROOT / 'rs_normalized')))
DESTINATION_DIR = PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/rs_indices'
TARGET_RESOLUTION = 0.1
NODATA_VALUE = 32767
PROCESS_COUNT = 2
GDAL_THREAD_COUNT = 2
WARP_MEMORY_MB = 512
gdal.UseExceptions()

def create_progress_bar(total):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return tqdm(total=total, desc='Resampling', unit='file', dynamic_ncols=True, smoothing=0.1)

def output_is_complete(path, width, height, geotransform):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
        if dataset is None:
            return False
        actual_geotransform = dataset.GetGeoTransform()
        structure_ok = dataset.RasterXSize == width and dataset.RasterYSize == height and (dataset.RasterCount == 1) and bool(dataset.GetProjectionRef()) and all((math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10) for actual, expected in zip(actual_geotransform, geotransform)))
        band = dataset.GetRasterBand(1)
        band_ok = structure_ok and band.DataType == gdal.GDT_Int16 and (band.GetNoDataValue() == NODATA_VALUE)
        dataset = None
        return band_ok
    except RuntimeError:
        return False

def resample_one(job):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    source_path, destination_path = job
    temporary_path = destination_path.with_name(f'{destination_path.stem}.part{destination_path.suffix}')
    try:
        gdal.SetConfigOption('GDAL_NUM_THREADS', str(GDAL_THREAD_COUNT))
        gdal.SetCacheMax(WARP_MEMORY_MB * 1024 * 1024)
        source_dataset = gdal.OpenEx(str(source_path), gdal.OF_RASTER | gdal.OF_READONLY)
        if source_dataset is None:
            raise RuntimeError('Cannot open source raster')
        if source_dataset.RasterCount != 1:
            raise RuntimeError(f'Expected one band; actual {source_dataset.RasterCount} items')
        source_band = source_dataset.GetRasterBand(1)
        if source_band.DataType != gdal.GDT_Int16:
            raise RuntimeError(f'Expected Int16; actual {gdal.GetDataTypeName(source_band.DataType)}')
        if source_band.GetNoDataValue() != NODATA_VALUE:
            raise RuntimeError(f'Expected NoData={NODATA_VALUE}; actual value: {source_band.GetNoDataValue()}')
        source_geotransform = source_dataset.GetGeoTransform()
        if not math.isclose(source_geotransform[2], 0.0, abs_tol=1e-12) or not math.isclose(source_geotransform[4], 0.0, abs_tol=1e-12) or source_geotransform[1] <= 0.0 or (source_geotransform[5] >= 0.0):
            raise RuntimeError('Source raster is not an unrotated north-up grid')
        left = source_geotransform[0]
        top = source_geotransform[3]
        source_right = left + source_dataset.RasterXSize * source_geotransform[1]
        source_bottom = top + source_dataset.RasterYSize * source_geotransform[5]
        target_width = math.ceil((source_right - left) / TARGET_RESOLUTION - 1e-10)
        target_height = math.ceil((top - source_bottom) / TARGET_RESOLUTION - 1e-10)
        target_right = left + target_width * TARGET_RESOLUTION
        target_bottom = top - target_height * TARGET_RESOLUTION
        target_geotransform = (left, TARGET_RESOLUTION, 0.0, top, 0.0, -TARGET_RESOLUTION)
        source_projection = source_dataset.GetProjectionRef()
        source_dataset = None
        if not source_projection:
            raise RuntimeError('Source raster has no CRS')
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if output_is_complete(destination_path, target_width, target_height, target_geotransform):
            return ('SKIP', str(source_path.relative_to(SOURCE_DIR)), 'Complete output already exists')
        temporary_path.unlink(missing_ok=True)
        options = gdal.WarpOptions(format='GTiff', outputBounds=(left, target_bottom, target_right, top), width=target_width, height=target_height, dstSRS=source_projection, resampleAlg='average', srcNodata=NODATA_VALUE, dstNodata=NODATA_VALUE, outputType=gdal.GDT_Int16, multithread=True, warpMemoryLimit=WARP_MEMORY_MB, warpOptions=(f'NUM_THREADS={GDAL_THREAD_COUNT}', 'UNIFIED_SRC_NODATA=NO', 'INIT_DEST=NO_DATA'), creationOptions=('TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6', f'NUM_THREADS={GDAL_THREAD_COUNT}', 'BIGTIFF=IF_SAFER'))
        result = gdal.Warp(str(temporary_path), str(source_path), options=options)
        if result is None:
            raise RuntimeError('gdal.Warp produced no output')
        result.FlushCache()
        result = None
        if not output_is_complete(temporary_path, target_width, target_height, target_geotransform):
            raise RuntimeError('Temporary output failed integrity checks')
        os.replace(temporary_path, destination_path)
        return ('OK', str(source_path.relative_to(SOURCE_DIR)), 'Complete')
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        return ('FAIL', str(source_path), str(error))
if __name__ == '__main__':
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {SOURCE_DIR}')
    source_files = raster_inputs(SOURCE_DIR)
    if not source_files:
        raise FileNotFoundError(f'No GeoTIFF files found in input directory: {SOURCE_DIR}')
    jobs = [(source_path, DESTINATION_DIR / source_path.relative_to(SOURCE_DIR).with_suffix('.tif')) for source_path in source_files]
    print(f'Pending checks/tasks: {len(jobs)} files')
    print(f'Parallel settings: {PROCESS_COUNT} processes; threads per process: {GDAL_THREAD_COUNT} GDAL threads')
    counts = {'OK': 0, 'SKIP': 0, 'FAIL': 0}
    failures = []
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        futures = [executor.submit(resample_one, job) for job in jobs]
        with create_progress_bar(len(futures)) as progress:
            for future in as_completed(futures):
                status, relative_path, message = future.result()
                counts[status] += 1
                if status == 'FAIL':
                    failures.append((relative_path, message))
                progress.set_postfix(ok=counts['OK'], skip=counts['SKIP'], fail=counts['FAIL'], refresh=False)
                progress.update(1)
    print(f"Processing complete; succeeded: {counts['OK']}; skipped: {counts['SKIP']}; failed: {counts['FAIL']}")
    if failures:
        failure_text = '\n'.join((f'{path}: {message}' for path, message in failures))
        raise RuntimeError(f'The following files failed:\n{failure_text}')
