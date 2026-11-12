"""Unify PV and NPV indicator directions for DSI construction."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
import shutil
from pathlib import Path
import numpy as np
from osgeo import gdal
from tqdm import tqdm
SOURCE_DIR = PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/fvc_original'
DESTINATION_DIR = PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/fvc_direction_unified'
COMPONENTS = ('PV', 'NPV', 'BS', 'IS', 'DA')
SCALE_MAX = 10000
NODATA = 32767
OVERWRITE = False
gdal.UseExceptions()
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(f'Input directory does not exist: {SOURCE_DIR}')
source_files = []
for component in COMPONENTS:
    component_dir = SOURCE_DIR / component
    if not component_dir.is_dir():
        raise FileNotFoundError(f'Missing component directory: {component_dir}')
    source_files.extend(((component, path) for path in component_dir.rglob('*.tif') if path.is_file()))
source_files.sort(key=lambda item: str(item[1]))
if not source_files:
    raise FileNotFoundError('No TIFF files found in PV and NPV directories')
driver = gdal.GetDriverByName('GTiff')
if driver is None:
    raise RuntimeError('GDAL GTiff driver is unavailable')
completed_count = 0
skipped_count = 0
for component, source_path in tqdm(source_files, desc='Unify PV/NPV indicator directions', unit='file', dynamic_ncols=True):
    relative_path = source_path.relative_to(SOURCE_DIR)
    destination_path = DESTINATION_DIR / relative_path
    if destination_path.exists() and (not OVERWRITE):
        skipped_count += 1
        continue
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if component not in ('PV', 'NPV'):
        shutil.copy2(source_path, destination_path)
        completed_count += 1
        continue
    temporary_path = destination_path.with_name(f'{destination_path.stem}.part{destination_path.suffix}')
    temporary_path.unlink(missing_ok=True)
    source_dataset = None
    source_band = None
    output_dataset = None
    output_band = None
    try:
        source_dataset = gdal.OpenEx(str(source_path), gdal.OF_RASTER | gdal.OF_READONLY)
        if source_dataset is None:
            raise RuntimeError('Cannot open input raster')
        if source_dataset.RasterCount != 1:
            raise RuntimeError(f'Single-band raster required; actual {source_dataset.RasterCount} bands')
        source_band = source_dataset.GetRasterBand(1)
        if source_band.DataType != gdal.GDT_Int16:
            raise RuntimeError(f'Int16 required; actual {gdal.GetDataTypeName(source_band.DataType)}')
        if source_band.GetNoDataValue() != NODATA:
            raise RuntimeError(f'Required NoData={NODATA}; actual value: {source_band.GetNoDataValue()}')
        width = source_dataset.RasterXSize
        height = source_dataset.RasterYSize
        geotransform = source_dataset.GetGeoTransform()
        projection = source_dataset.GetProjectionRef()
        if not projection:
            raise RuntimeError('Input raster has no CRS')
        output_dataset = driver.Create(str(temporary_path), width, height, 1, gdal.GDT_Int16, options=('TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=IF_SAFER'))
        if output_dataset is None:
            raise RuntimeError('Cannot create temporary output raster')
        output_dataset.SetGeoTransform(geotransform)
        output_dataset.SetProjection(projection)
        output_dataset.SetMetadata(source_dataset.GetMetadata())
        output_dataset.SetMetadataItem('DIRECTION_NORMALIZATION', '10000 - original')
        output_dataset.SetMetadataItem('SOURCE_FILE', str(source_path))
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(NODATA)
        output_band.SetDescription(source_band.GetDescription() or component)
        output_band.SetColorInterpretation(source_band.GetColorInterpretation())
        band_metadata = {key: value for key, value in source_band.GetMetadata().items() if not key.startswith('STATISTICS_')}
        output_band.SetMetadata(band_metadata)
        block_width, block_height = source_band.GetBlockSize()
        for y_offset in range(0, height, block_height):
            y_size = min(block_height, height - y_offset)
            for x_offset in range(0, width, block_width):
                x_size = min(block_width, width - x_offset)
                source_data = source_band.ReadAsArray(x_offset, y_offset, x_size, y_size)
                if source_data is None:
                    raise RuntimeError(f'Cannot read tile at x={x_offset}, y={y_offset}')
                valid = source_data != NODATA
                if np.any(valid):
                    valid_values = source_data[valid]
                    valid_min = int(valid_values.min())
                    valid_max = int(valid_values.max())
                    if valid_min < 0 or valid_max > SCALE_MAX:
                        raise RuntimeError(f'Valid pixels outside the expected 0-10000 range; minimum={valid_min}; maximum={valid_max}')
                normalized_data = np.full(source_data.shape, NODATA, dtype=np.int16)
                normalized_data[valid] = (SCALE_MAX - source_data[valid].astype(np.int32)).astype(np.int16)
                output_band.WriteArray(normalized_data, x_offset, y_offset)
        output_band.FlushCache()
        output_dataset.FlushCache()
        output_band = None
        output_dataset = None
        source_band = None
        source_dataset = None
        check_dataset = gdal.OpenEx(str(temporary_path), gdal.OF_RASTER | gdal.OF_READONLY)
        if check_dataset is None:
            raise RuntimeError('Cannot reopen temporary output for verification')
        check_band = check_dataset.GetRasterBand(1)
        output_valid = check_dataset.RasterXSize == width and check_dataset.RasterYSize == height and (check_dataset.RasterCount == 1) and (check_band.DataType == gdal.GDT_Int16) and (check_band.GetNoDataValue() == NODATA) and (check_dataset.GetGeoTransform() == geotransform) and bool(check_dataset.GetProjectionRef())
        check_band = None
        check_dataset = None
        if not output_valid:
            raise RuntimeError('Temporary output failed structural checks')
        os.replace(temporary_path, destination_path)
        completed_count += 1
    except Exception as error:
        output_band = None
        output_dataset = None
        source_band = None
        source_dataset = None
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f'Processing failed: {source_path}\n{error}') from error
print(f'Processing completed; generated {completed_count} files; existing files skipped: {skipped_count} files; output directory: {DESTINATION_DIR}')
