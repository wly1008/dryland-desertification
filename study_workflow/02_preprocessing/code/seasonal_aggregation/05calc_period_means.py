"""Aggregate monthly indicators over October-April non-growing seasons."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import os
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm
DATASETS = ((PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/rs_indices', PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/rs_indices'), (PROJECT_ROOT / '02_preprocessing/data/monthly_0p1/fvc_direction_unified', PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc'))
PERIODS = ({'name': 'non-growing_season', 'key': 'non-growing_season', 'output_relative': remap_path('.'), 'start_year': 2001, 'end_year': 2021, 'months': ((0, 10), (0, 11), (0, 12), (1, 1), (1, 2), (1, 3), (1, 4)), 'start_month': 10, 'end_year_offset': 1, 'end_month': 4, 'definition': 'October through April of the following year'},)
OUTPUT_NODATA = 32767.0
OUTPUT_DTYPE = 'float32'
OVERWRITE = 0

def create_progress_bar(total):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return tqdm(total=total, desc='Period means', unit='file', dynamic_ncols=True, smoothing=1)
jobs = []
for source_root, normalization_root in DATASETS:
    if not source_root.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {source_root}')
    variable_dirs = sorted((path for path in source_root.iterdir() if path.is_dir()))
    if not variable_dirs:
        raise FileNotFoundError(f'No variable subdirectories in input directory: {source_root}')
    for period in PERIODS:
        for variable_dir in variable_dirs:
            variable = variable_dir.name
            for start_year in range(period['start_year'], period['end_year'] + 1):
                source_paths = []
                for year_offset, month in period['months']:
                    source_year = start_year + year_offset
                    source_path = variable_dir / str(source_year) / f'{variable}_{source_year}_{month:02d}.tif'
                    if not source_path.is_file():
                        raise FileNotFoundError(f"{period['name']}Missing monthly file: {source_path}")
                    source_paths.append(source_path)
                period_start = f"{start_year}-{period['start_month']:02d}"
                period_end = f"{start_year + period['end_year_offset']}-{period['end_month']:02d}"
                output_path = normalization_root / period['output_relative'] / variable / f'{variable}_{start_year}.tif'
                jobs.append((period, variable, start_year, period_start, period_end, source_paths, output_path))
print(f'Found {len(jobs)} complete variable-period mean tasks.')
written = 0
skipped = 0
progress = create_progress_bar(len(jobs))
try:
    for period, variable, start_year, period_start, period_end, source_paths, output_path in jobs:
        progress.set_description(f"{period['name']} {variable} {start_year}")
        if output_path.exists() and (not OVERWRITE):
            try:
                with rasterio.open(output_path) as existing:
                    tags = existing.tags()
                    output_is_complete = existing.count == 1 and existing.dtypes[0] == OUTPUT_DTYPE and (existing.nodata == OUTPUT_NODATA) and (tags.get('STATISTIC') == 'mean') and (tags.get('PERIOD_KIND') == period['key']) and (tags.get('PERIOD_START') == period_start) and (tags.get('PERIOD_END') == period_end)
            except rasterio.errors.RasterioError:
                output_is_complete = False
            if output_is_complete:
                skipped += 1
                progress.set_postfix(written=written, skipped=skipped, refresh=False)
                progress.update(1)
                continue
            raise RuntimeError(f'Existing output failed integrity checks; inspect or set OVERWRITE=True: {output_path}')
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(path)) for path in source_paths]
            reference = sources[0]
            if reference.count != 1:
                raise RuntimeError(f'Source raster is not single-band: {source_paths[0]}')
            if reference.crs is None:
                raise RuntimeError(f'Source raster has no CRS: {source_paths[0]}')
            for source_path, source in zip(source_paths[1:], sources[1:]):
                if source.count != 1 or source.width != reference.width or source.height != reference.height or (source.crs != reference.crs) or (not source.transform.almost_equals(reference.transform)):
                    raise RuntimeError(f'Rasters within a period differ in bands, dimensions, CRS, or grid: {source_path}')
            monthly_stack = np.ma.stack([source.read(1, masked=True) for source in sources], axis=0)
            period_mean = np.ma.mean(monthly_stack, axis=0)
            output_array = period_mean.filled(OUTPUT_NODATA).astype(OUTPUT_DTYPE)
            output_profile = reference.profile.copy()
            output_profile.update(driver='GTiff', count=1, dtype=OUTPUT_DTYPE, nodata=OUTPUT_NODATA, tiled=True, blockxsize=256, blockysize=256, compress='deflate', predictor=3, zlevel=6, BIGTIFF='IF_SAFER')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f'{output_path.stem}.part{output_path.suffix}')
        temporary_path.unlink(missing_ok=True)
        with rasterio.open(temporary_path, 'w', **output_profile) as destination:
            destination.write(output_array, 1)
            destination.set_band_description(1, variable)
            destination.update_tags(STATISTIC='mean', PERIOD_KIND=period['key'], PERIOD_DEFINITION=period['definition'], PERIOD_START=period_start, PERIOD_END=period_end, SOURCE_COUNT=str(len(source_paths)), NODATA_POLICY='ignore nodata; nodata only when all source months are nodata')
        with rasterio.open(temporary_path) as check:
            if check.count != 1 or check.width != output_profile['width'] or check.height != output_profile['height'] or (check.dtypes[0] != np.dtype(output_profile['dtype']).name) or (check.nodata != OUTPUT_NODATA) or (check.crs != output_profile['crs']) or (not check.transform.almost_equals(output_profile['transform'])) or (check.tags().get('STATISTIC') != 'mean') or (check.tags().get('PERIOD_START') != period_start) or (check.tags().get('PERIOD_END') != period_end):
                raise RuntimeError(f'Temporary output failed integrity checks: {temporary_path}')
        os.replace(temporary_path, output_path)
        written += 1
        progress.set_postfix(written=written, skipped=skipped, refresh=False)
        progress.update(1)
finally:
    progress.close()
print(f'Processing completed; wrote {written} outputs; complete existing outputs skipped: {skipped} items.')
