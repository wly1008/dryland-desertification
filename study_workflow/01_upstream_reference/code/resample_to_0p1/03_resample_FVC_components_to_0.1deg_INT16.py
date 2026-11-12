"""Resample vegetation-cover components to the 0.1-degree INT16 grid."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, raster_source, raster_inputs
import gc
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from osgeo import gdal
from tqdm import tqdm
SOURCE_DIR = VRT_ROOT / 'sma'
DESTINATION_DIR = PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/fvc_original'
COMPONENTS = ('PV', 'NPV', 'BS', 'IS', 'DA')
TARGET_RESOLUTION = 0.1
SOURCE_NODATA = 32767
DESTINATION_NODATA = 32767
PROCESS_COUNT = 1
GDAL_THREAD_COUNT = 2
WARP_MEMORY_MB = 2048
GDAL_CACHE_MB = 8192
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10
ALGORITHM_VERSION = 'invalid_or_allzero_to_DA10000_average_first4zero_nodata_v4'
TEST_ONE_MONTH = False
TEST_YEAR = '2001'
TEST_MONTH = '01'
START_YEAR = 2001
END_YEAR = 2022
gdal.UseExceptions()

def create_progress_bar(total):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return tqdm(total=total, desc='FVC resampling', unit='month', dynamic_ncols=True, smoothing=0.1)

def find_source_files(start_year, end_year):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise TypeError('START_YEAR and END_YEAR must be integers')
    if start_year > end_year:
        raise ValueError('START_YEAR must not exceed END_YEAR')
    selected = []
    for path in raster_inputs(SOURCE_DIR):
        name_match = re.fullmatch('SMA_(\\d{4})_(\\d{2})', path.stem)
        if name_match is None:
            continue
        year = int(name_match.group(1))
        month = int(name_match.group(2))
        if start_year <= year <= end_year and 1 <= month <= 12:
            selected.append(path)
    return sorted(selected)

def build_replacement_vrt_xml(source_path, width, height, geotransform, projection):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    pixel_function_code = 'import numpy as np\ndef replace_invalid(in_ar, out_ar, xoff, yoff, xsize, ysize,\n                    raster_xsize, raster_ysize, buf_radius, gt, **kwargs):\n    invalid = in_ar[0] == 32767\n    all_zero = in_ar[0] == 0\n    for array in in_ar[1:]:\n        invalid |= array == 32767\n        all_zero &= array == 0\n    invalid |= all_zero\n    target = int(kwargs["target"])\n    out_ar[:] = in_ar[target]\n    out_ar[invalid] = 10000 if target == 4 else 0\n'
    escaped_source = escape(str(source_path))
    sources = ''.join((f'\n        <SimpleSource>\n          <SourceFilename relativeToVRT="0">{escaped_source}</SourceFilename>\n          <SourceBand>{source_band}</SourceBand>\n          <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n          <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n        </SimpleSource>' for source_band in range(1, len(COMPONENTS) + 1)))
    bands = ''.join((f'\n      <VRTRasterBand dataType="Int16" band="{band_index}"\n                     blockXSize="256" blockYSize="256"\n                     subClass="VRTDerivedRasterBand">\n        <Description>{component}</Description>\n        <PixelFunctionType>replace_invalid</PixelFunctionType>\n        <PixelFunctionLanguage>Python</PixelFunctionLanguage>\n        <PixelFunctionArguments target="{band_index - 1}"/>\n        <PixelFunctionCode><![CDATA[{pixel_function_code}]]></PixelFunctionCode>\n        {sources}\n      </VRTRasterBand>' for band_index, component in enumerate(COMPONENTS, start=1)))
    geotransform_text = ','.join((f'{value:.17g}' for value in geotransform))
    return f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n      <SRS>{escape(projection)}</SRS>\n      <GeoTransform>{geotransform_text}</GeoTransform>\n      {bands}\n    </VRTDataset>'

def output_is_complete(path, width, height, geotransform, component):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
        if dataset is None:
            return False
        actual_geotransform = dataset.GetGeoTransform()
        structure_ok = dataset.RasterXSize == width and dataset.RasterYSize == height and (dataset.RasterCount == 1) and bool(dataset.GetProjectionRef()) and (dataset.GetMetadataItem('ALGORITHM_VERSION') == ALGORITHM_VERSION) and all((math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10) for actual, expected in zip(actual_geotransform, geotransform)))
        band = dataset.GetRasterBand(1)
        band_ok = structure_ok and band.DataType == gdal.GDT_Int16 and (band.GetNoDataValue() == DESTINATION_NODATA) and (band.GetDescription() == component)
        band = None
        dataset = None
        return band_ok
    except RuntimeError:
        return False

def _resample_month_once(source_path):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    results = []
    temporary_paths = []
    try:
        gdal.SetConfigOption('GDAL_NUM_THREADS', str(GDAL_THREAD_COUNT))
        gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
        gdal.SetCacheMax(GDAL_CACHE_MB * 1024 * 1024)
        name_match = re.fullmatch('SMA_(\\d{4})_(\\d{2})', source_path.stem)
        if name_match is None:
            raise RuntimeError(f'Filename does not match SMA_YYYY_MM: {source_path.name}')
        year, month = name_match.groups()
        source_dataset = gdal.OpenEx(str(source_path), gdal.OF_RASTER | gdal.OF_READONLY)
        if source_dataset is None:
            raise RuntimeError('Cannot open source VRT')
        if source_dataset.RasterCount != len(COMPONENTS):
            raise RuntimeError(f'Expected {len(COMPONENTS)} bands; actual: {source_dataset.RasterCount} items')
        for band_index, component in enumerate(COMPONENTS, start=1):
            source_band = source_dataset.GetRasterBand(band_index)
            if source_band.GetDescription() != component:
                raise RuntimeError(f"Band {band_index} expected {component}; actual value: {source_band.GetDescription() or 'Empty description'}")
            if source_band.DataType != gdal.GDT_Int16:
                raise RuntimeError(f'{component} expected Int16; actual {gdal.GetDataTypeName(source_band.DataType)}')
            if source_band.GetNoDataValue() != SOURCE_NODATA:
                raise RuntimeError(f'{component} expected NoData={SOURCE_NODATA}; actual value: {source_band.GetNoDataValue()}')
        source_geotransform = source_dataset.GetGeoTransform()
        if not math.isclose(source_geotransform[2], 0.0, abs_tol=1e-12) or not math.isclose(source_geotransform[4], 0.0, abs_tol=1e-12) or source_geotransform[1] <= 0.0 or (source_geotransform[5] >= 0.0):
            raise RuntimeError('Source raster is not an unrotated north-up grid')
        left = source_geotransform[0]
        top = source_geotransform[3]
        source_width = source_dataset.RasterXSize
        source_height = source_dataset.RasterYSize
        source_right = left + source_width * source_geotransform[1]
        source_bottom = top + source_height * source_geotransform[5]
        target_width = math.ceil((source_right - left) / TARGET_RESOLUTION - 1e-10)
        target_height = math.ceil((top - source_bottom) / TARGET_RESOLUTION - 1e-10)
        target_right = left + target_width * TARGET_RESOLUTION
        target_bottom = top - target_height * TARGET_RESOLUTION
        target_geotransform = (left, TARGET_RESOLUTION, 0.0, top, 0.0, -TARGET_RESOLUTION)
        source_projection = source_dataset.GetProjectionRef()
        source_dataset = None
        source_band = None
        if not source_projection:
            raise RuntimeError('Source VRT has no CRS')
        destinations = [(component, DESTINATION_DIR / component / year / f'{component}_{year}_{month}.tif') for component in COMPONENTS]
        completion = {component: output_is_complete(destination_path, target_width, target_height, target_geotransform, component) for component, destination_path in destinations}
        if all(completion.values()):
            return [('SKIP', str(destination_path), 'Complete Int16 output already exists') for component, destination_path in destinations]
        replacement_vrt_xml = build_replacement_vrt_xml(source_path, source_width, source_height, source_geotransform, source_projection)
        replacement_dataset = gdal.OpenEx(replacement_vrt_xml, gdal.OF_RASTER | gdal.OF_READONLY)
        if replacement_dataset is None:
            raise RuntimeError('Cannot create the invalid-region replacement VRT')
        warp_options = gdal.WarpOptions(format='MEM', outputBounds=(left, target_bottom, target_right, top), width=target_width, height=target_height, dstSRS=source_projection, resampleAlg='average', dstNodata=DESTINATION_NODATA, outputType=gdal.GDT_Int16, overviewLevel='NONE', multithread=True, warpMemoryLimit=WARP_MEMORY_MB, warpOptions=(f'NUM_THREADS={GDAL_THREAD_COUNT}', 'INIT_DEST=NO_DATA'))
        warped_dataset = gdal.Warp('', replacement_dataset, options=warp_options)
        if warped_dataset is None:
            raise RuntimeError('gdal.Warp produced no in-memory output')
        replacement_dataset = None
        warped_data = warped_dataset.ReadAsArray()
        if warped_data is None:
            raise RuntimeError('Cannot read in-memory resampling output')
        uncovered = np.any(warped_data == DESTINATION_NODATA, axis=0)
        filled_uncovered_pixels = int(np.count_nonzero(uncovered))
        warped_data[:4, uncovered] = 0
        warped_data[4, uncovered] = 10000
        warped_dataset = None
        first_four_zero = np.all(warped_data[:4] == 0, axis=0)
        first_four_zero_count = int(np.count_nonzero(first_four_zero))
        warped_data[:, first_four_zero] = DESTINATION_NODATA
        driver = gdal.GetDriverByName('GTiff')
        for band_index, (component, destination_path) in enumerate(destinations, start=1):
            if completion[component]:
                results.append(('SKIP', str(destination_path), 'Complete Int16 output already exists'))
                continue
            temporary_path = destination_path.with_name(f'{destination_path.stem}.part{destination_path.suffix}')
            temporary_paths.append(temporary_path)
            try:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path.unlink(missing_ok=True)
                output_dataset = driver.Create(str(temporary_path), target_width, target_height, 1, gdal.GDT_Int16, options=('TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6', f'NUM_THREADS={GDAL_THREAD_COUNT}', 'BIGTIFF=IF_SAFER'))
                if output_dataset is None:
                    raise RuntimeError('Cannot create temporary output')
                output_dataset.SetGeoTransform(target_geotransform)
                output_dataset.SetProjection(source_projection)
                output_dataset.SetMetadataItem('ALGORITHM_VERSION', ALGORITHM_VERSION)
                output_dataset.SetMetadataItem('INVALID_REPLACEMENT', 'PV=0,NPV=0,BS=0,IS=0,DA=10000')
                output_dataset.SetMetadataItem('FIRST_FOUR_ZERO_RULE', f'PV=NPV=BS=IS=0 => NoData={DESTINATION_NODATA}')
                output_band = output_dataset.GetRasterBand(1)
                output_band.SetNoDataValue(DESTINATION_NODATA)
                output_band.SetDescription(component)
                output_band.WriteArray(warped_data[band_index - 1])
                output_band.FlushCache()
                output_dataset.FlushCache()
                output_band = None
                output_dataset = None
                if not output_is_complete(temporary_path, target_width, target_height, target_geotransform, component):
                    raise RuntimeError('Temporary output failed integrity checks')
                os.replace(temporary_path, destination_path)
                results.append(('OK', str(destination_path), f'Complete; replace invalid/all-zero five-component pixels with [0,0,0,0,10000] before averaging; added edge target pixels: {filled_uncovered_pixels}; first-four-components-all-zero areas set to NoData: {first_four_zero_count}'))
            except Exception as error:
                if output_is_complete(temporary_path, target_width, target_height, target_geotransform, component):
                    results.append(('FAIL', str(destination_path), f'{error}; complete new output retained at {temporary_path}'))
                else:
                    temporary_path.unlink(missing_ok=True)
                    results.append(('FAIL', str(destination_path), str(error)))
        return results
    except Exception as error:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        return [('FAIL', str(source_path), f'Entire month left unprocessed: {error}') for _ in COMPONENTS]

def resample_month(source_path):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    last_results = None
    total_attempts = MAX_RETRIES + 1
    for attempt_index in range(total_attempts):
        last_results = _resample_month_once(source_path)
        failed_results = [result for result in last_results if result[0] == 'FAIL']
        if not failed_results:
            if attempt_index > 0:
                attempt_number = attempt_index + 1
                last_results = [(status, output_path, f'{message}; attempt {attempt_number} succeeded') for status, output_path, message in last_results]
            return last_results
        if attempt_index < MAX_RETRIES:
            attempt_number = attempt_index + 1
            first_error = failed_results[0][2].replace('\n', ' ')
            print(f'\n{source_path.stem}: attempt {attempt_number}/{total_attempts} failed; retry after {RETRY_DELAY_SECONDS} seconds: {first_error}', flush=True)
            gdal.ErrorReset()
            gc.collect()
            time.sleep(RETRY_DELAY_SECONDS)
    return [(status, output_path, f'{message}; initial attempt and {MAX_RETRIES} automatic retries all failed' if status == 'FAIL' else message) for status, output_path, message in last_results]
if __name__ == '__main__':
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {SOURCE_DIR}')
    if TEST_ONE_MONTH:
        test_path = SOURCE_DIR / TEST_YEAR / f'SMA_{TEST_YEAR}_{TEST_MONTH}.vrt'
        if not test_path.is_file():
            raise FileNotFoundError(f'Test month does not exist: {test_path}')
        source_files = [test_path]
        print(f'Test mode: process only {TEST_YEAR}-{TEST_MONTH}')
    else:
        source_files = find_source_files(START_YEAR, END_YEAR)
        if not source_files:
            raise FileNotFoundError(f'{START_YEAR}–{END_YEAR} year has no monthly VRTs: {SOURCE_DIR}')
        print(f'Full mode: years {START_YEAR}–{END_YEAR}; process {len(source_files)} months')
    print(f'Parallel settings: {PROCESS_COUNT} processes; threads per process: {GDAL_THREAD_COUNT} GDAL threads')
    print(f'Retry settings: at most {MAX_RETRIES} attempts; interval {RETRY_DELAY_SECONDS} seconds')
    print(f'Memory settings: Warp {WARP_MEMORY_MB} MB; GDAL cache {GDAL_CACHE_MB} MB')
    counts = {'OK': 0, 'SKIP': 0, 'FAIL': 0}
    failures = []
    failure_keys = set()
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        futures = [executor.submit(resample_month, path) for path in source_files]
        with create_progress_bar(len(futures)) as progress:
            for future in as_completed(futures):
                month_results = future.result()
                for status, output_path, message in month_results:
                    counts[status] += 1
                    if status == 'FAIL':
                        failure_key = (output_path, message)
                        if failure_key not in failure_keys:
                            failure_keys.add(failure_key)
                            failures.append(failure_key)
                progress.set_postfix(ok=counts['OK'], skip=counts['SKIP'], fail=counts['FAIL'], refresh=False)
                progress.update(1)
    print(f"Processing complete; succeeded: {counts['OK']}; skipped: {counts['SKIP']}; failed: {counts['FAIL']}")
    if failures:
        failure_text = '\n'.join((f'{path}: {message}' for path, message in failures))
        raise RuntimeError(f'The following files failed:\n{failure_text}')
