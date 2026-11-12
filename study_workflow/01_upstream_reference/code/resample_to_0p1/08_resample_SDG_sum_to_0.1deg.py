"""Resample SDG comparison products to the 0.1-degree grid."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, raster_source, raster_inputs, same_source
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from align_raster import align_raster
SOURCE_DIR = VRT_ROOT / 'auxiliary_sources'
OUTPUT_ROOT = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1/resample_0.1deg_sum'
REFERENCE_RASTER = PROJECT_ROOT / '02_preprocessing/data/auxiliary/masks_0p1/arid_region_mask_a.tif'
SOURCE_NAMES = ('TrendsEarth_SDG15.3.1_2000-2023_FAO-WOCAT.tif', 'TrendsEarth_SDG15.3.1_2000-2023_JRC.tif', 'TrendsEarth_SDG15.3.1_2000-2023_Trends.Earth.tif')
TARGET_BANDS = {'SDG Indicator 15.3.1 for 2004-2019': 'SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'}
OUTPUT_DTYPE = 'float32'
OUTPUT_NODATA = -32768.0
MEMORY_LIMIT_MIB = 512
NUM_THREADS = 1
TILE_WIDTH = 256
TILE_HEIGHT = 128
GDAL_VRT_DATA_TYPES = {'uint8': 'Byte', 'int8': 'Int8', 'uint16': 'UInt16', 'int16': 'Int16', 'uint32': 'UInt32', 'int32': 'Int32', 'uint64': 'UInt64', 'int64': 'Int64', 'float32': 'Float32', 'float64': 'Float64'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('resample_sdg')
started_at = time.perf_counter()
if not REFERENCE_RASTER.is_file():
    raise FileNotFoundError(f'Reference mask does not exist: {REFERENCE_RASTER}')
with rasterio.open(REFERENCE_RASTER) as reference:
    if reference.count < 1:
        raise RuntimeError('Reference mask has no usable bands')
    if reference.crs is None:
        raise RuntimeError('Reference mask has no CRS')
    if not (math.isclose(reference.transform.a, 0.1, abs_tol=1e-12) and math.isclose(reference.transform.e, -0.1, abs_tol=1e-12)):
        raise RuntimeError(f'Reference-mask resolution is not 0.1 degrees: {reference.transform.a}, {reference.transform.e}')
    if reference.nodata is None:
        raise RuntimeError('Reference mask has no NoData; apply_mask=True cannot identify masked-out cells')
    reference_crs = reference.crs
    reference_transform = reference.transform
    reference_width = reference.width
    reference_height = reference.height
    reference_dtype = reference.dtypes[0]
    reference_nodata = reference.nodata
    reference_data = reference.read(1)
active_windows = []
for row_offset in range(0, reference_height, TILE_HEIGHT):
    tile_height = min(TILE_HEIGHT, reference_height - row_offset)
    for column_offset in range(0, reference_width, TILE_WIDTH):
        tile_width = min(TILE_WIDTH, reference_width - column_offset)
        window = Window(column_offset, row_offset, tile_width, tile_height)
        row_slice, column_slice = window.toslices()
        if np.any(reference_data[row_slice, column_slice] != reference_nodata):
            active_windows.append(window)
logger.info('Target grid: %d columns by %d rows; resolution 0.1 degrees; valid tiles=%d; reference=%s', reference_width, reference_height, len(active_windows), REFERENCE_RASTER)
missing_sources = [raster_source(SOURCE_DIR, Path(source_name).stem) for source_name in SOURCE_NAMES if not raster_source(SOURCE_DIR, Path(source_name).stem).is_file()]
if missing_sources:
    raise FileNotFoundError('Source rasters do not exist:\n' + '\n'.join((str(path) for path in missing_sources)))
completed_outputs = []
for source_number, source_name in enumerate(SOURCE_NAMES, start=1):
    source_path = raster_source(SOURCE_DIR, Path(source_name).stem)
    output_directory = OUTPUT_ROOT / source_path.stem
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info('[%d/%d] Check source raster: %s', source_number, len(SOURCE_NAMES), source_path)
    with rasterio.open(source_path) as source_dataset:
        if source_dataset.crs is None:
            raise RuntimeError(f'Source raster has no CRS: {source_path}')
        descriptions_to_indexes = {}
        for band_index, description in enumerate(source_dataset.descriptions, start=1):
            if description:
                if description in descriptions_to_indexes:
                    raise RuntimeError(f'Source raster has duplicate band descriptions: {source_path}；{description!r}')
                descriptions_to_indexes[description] = band_index
        source_width = source_dataset.width
        source_height = source_dataset.height
        source_crs_wkt = source_dataset.crs.to_wkt()
        source_geotransform = source_dataset.transform.to_gdal()
        source_dtypes = source_dataset.dtypes
        source_nodatavals = source_dataset.nodatavals
        source_block_shapes = source_dataset.block_shapes
    missing_descriptions = [description for description in TARGET_BANDS if description not in descriptions_to_indexes]
    if missing_descriptions:
        raise RuntimeError(f'Requested source band is missing: {source_path}\n' + '\n'.join(missing_descriptions))
    with tempfile.TemporaryDirectory(prefix='sdg_selected_band_') as temp_dir:
        temporary_directory = remap_path(temp_dir)
        for band_number, (description, output_name) in enumerate(TARGET_BANDS.items(), start=1):
            band_index = descriptions_to_indexes[description]
            vrt_path = temporary_directory / f'band_{band_index}.vrt'
            output_path = output_directory / output_name
            logger.info('  [%d/%d] Band %d: %s', band_number, len(TARGET_BANDS), band_index, description)
            existing_complete = False
            if output_path.is_file():
                try:
                    with rasterio.open(output_path) as existing:
                        existing_data = existing.read(1)
                        existing_valid = existing_data != existing.nodata
                        existing_tags = existing.tags()
                        existing_complete = existing.count == 1 and existing.width == reference_width and (existing.height == reference_height) and (existing.crs == reference_crs) and existing.transform.almost_equals(reference_transform) and (existing.dtypes[0] == OUTPUT_DTYPE) and (existing.nodata == OUTPUT_NODATA) and (existing.descriptions[0] == description) and (existing_tags.get('RESAMPLING') == 'sum') and same_source(existing_tags.get('SOURCE_RASTER'), source_path) and (existing_tags.get('SOURCE_BAND') == str(band_index)) and (existing_tags.get('APPLY_REFERENCE_MASK') == 'True') and np.isfinite(existing_data[existing_valid]).all() and (not np.any(existing_valid & (reference_data == reference_nodata)))
                except (OSError, rasterio.errors.RasterioError):
                    existing_complete = False
            if existing_complete:
                completed_outputs.append(output_path)
                logger.info('  Skip complete existing output: %s', output_path)
                continue
            if output_path.exists():
                logger.warning('  Existing output is incomplete; regenerating: %s', output_path)
            source_dtype = source_dtypes[band_index - 1]
            if source_dtype not in GDAL_VRT_DATA_TYPES:
                raise RuntimeError(f'VRT does not support source data type: {source_path}，{source_dtype}')
            source_nodata = source_nodatavals[band_index - 1]
            if source_nodata is None:
                raise RuntimeError(f'Target source band has no NoData: {source_path}; band {band_index}')
            block_height, block_width = source_block_shapes[band_index - 1]
            geotransform_text = ','.join((f'{value:.17g}' for value in source_geotransform))
            vrt_xml = f'<VRTDataset rasterXSize="{source_width}" rasterYSize="{source_height}">\n  <SRS>{escape(source_crs_wkt)}</SRS>\n  <GeoTransform>{geotransform_text}</GeoTransform>\n  <VRTRasterBand dataType="{GDAL_VRT_DATA_TYPES[source_dtype]}" band="1"\n                 blockXSize="{block_width}" blockYSize="{block_height}">\n    <NoDataValue>{source_nodata:.17g}</NoDataValue>\n    <Description>{escape(description)}</Description>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{escape(str(source_path))}</SourceFilename>\n      <SourceBand>{band_index}</SourceBand>\n      <SourceProperties RasterXSize="{source_width}" RasterYSize="{source_height}"\n                        DataType="{GDAL_VRT_DATA_TYPES[source_dtype]}"\n                        BlockXSize="{block_width}" BlockYSize="{block_height}"/>\n      <SrcRect xOff="0" yOff="0" xSize="{source_width}" ySize="{source_height}"/>\n      <DstRect xOff="0" yOff="0" xSize="{source_width}" ySize="{source_height}"/>\n      <NODATA>{source_nodata:.17g}</NODATA>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>\n'
            vrt_path.write_text(vrt_xml, encoding='utf-8')
            output_profile = {'driver': 'GTiff', 'width': reference_width, 'height': reference_height, 'count': 1, 'dtype': OUTPUT_DTYPE, 'crs': reference_crs, 'transform': reference_transform, 'nodata': OUTPUT_NODATA, 'compress': 'DEFLATE', 'predictor': 3, 'zlevel': 6, 'tiled': True, 'blockxsize': 256, 'blockysize': 256, 'BIGTIFF': 'IF_SAFER'}
            file_descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output_path.stem}.', suffix='.tmp.tif', dir=output_directory)
            os.close(file_descriptor)
            temporary_output = remap_path(temporary_name)
            try:
                with rasterio.open(temporary_output, 'w', **output_profile) as output:
                    output.write(np.full((reference_height, reference_width), OUTPUT_NODATA, dtype=OUTPUT_DTYPE), 1)
                    for tile_number, window in enumerate(active_windows, start=1):
                        row_slice, column_slice = window.toslices()
                        mask_block = reference_data[row_slice, column_slice]
                        tile_transform = window_transform(window, reference_transform)
                        logger.info('    Tile %d/%d: columns %d-%d, rows %d-%d', tile_number, len(active_windows), int(window.col_off), int(window.col_off + window.width - 1), int(window.row_off), int(window.row_off + window.height - 1))
                        with MemoryFile() as reference_memory_file:
                            with reference_memory_file.open(driver='GTiff', width=int(window.width), height=int(window.height), count=1, dtype=reference_dtype, crs=reference_crs, transform=tile_transform, nodata=reference_nodata) as tile_reference:
                                tile_reference.write(mask_block, 1)
                                tile_array, _ = align_raster(source=vrt_path, reference=tile_reference, out_path=None, nodata=OUTPUT_NODATA, dtype=OUTPUT_DTYPE, resampling=Resampling.sum, apply_mask=True, mask_band=1, mem_limit=MEMORY_LIMIT_MIB, num_threads=NUM_THREADS)
                        output.write(tile_array[0], 1, window=window)
                    output.set_band_description(1, description)
                    output.update_tags(RESAMPLING='sum', REFERENCE_RASTER=str(REFERENCE_RASTER), SOURCE_RASTER=str(source_path), SOURCE_BAND=str(band_index), APPLY_REFERENCE_MASK='True', PROCESSING_LAYOUT=f'windowed_{TILE_WIDTH}x{TILE_HEIGHT}')
                os.replace(temporary_output, output_path)
            except Exception:
                temporary_output.unlink(missing_ok=True)
                raise
            with rasterio.open(output_path) as output:
                if output.count != 1 or output.width != reference_width or output.height != reference_height or (output.crs != reference_crs) or (not output.transform.almost_equals(reference_transform)) or (output.dtypes[0] != OUTPUT_DTYPE) or (output.nodata != OUTPUT_NODATA):
                    raise RuntimeError(f'Output structure validation failed: {output_path}')
            completed_outputs.append(output_path)
            logger.info('  Written: %s', output_path)
elapsed_seconds = time.perf_counter() - started_at
logger.info('Processing complete: %d outputs in %.1f minutes', len(completed_outputs), elapsed_seconds / 60.0)
for completed_output in completed_outputs:
    logger.info('Result: %s', completed_output)
