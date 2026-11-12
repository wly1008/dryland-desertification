"""Align MODIS land cover to the 0.1-degree grid using nearest-neighbour resampling."""
from __future__ import annotations
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, raster_inputs
import argparse
import os
import re
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
DEFAULT_SOURCE_DIR = VRT_ROOT / 'auxiliary_sources/land_cover_native'
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / '02_preprocessing/data/auxiliary/masks_0p1/arid_region_mask_a.tif'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/land_cover_0p1'
FILE_PATTERN = 'MCD12C1_LULC_*.tif'
YEAR_PATTERN = re.compile('^MCD12C1_LULC_(\\d{4})\\.(?:tif|tiff|vrt)$')
STUDY_YEARS = tuple(range(2001, 2022))
OUTPUT_NODATA = 255
OVERVIEW_LEVELS = (2, 4, 8, 16, 32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Align annual MODIS LULC rasters to the 0.1-degree arid-region grid.')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing outputs. Without this option, valid outputs are skipped.')
    parser.add_argument('--num-threads', type=int, default=4, help='Number of GDAL warp threads (default: 4).')
    return parser.parse_args()

def collect_sources(source_dir: Path) -> list[tuple[int, Path]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f'Source directory does not exist: {source_dir}')
    sources: list[tuple[int, Path]] = []
    for path in sorted(raster_inputs(source_dir, 'MCD12C1_LULC_*')):
        match = YEAR_PATTERN.match(path.name)
        if match and int(match.group(1)) in STUDY_YEARS:
            sources.append((int(match.group(1)), path))
    if not sources:
        raise FileNotFoundError(f'No files matching {FILE_PATTERN!r} were found in {source_dir}')
    years = [year for year, _ in sources]
    if len(years) != len(set(years)):
        raise ValueError(f'Duplicate years found: {years}')
    expected_years = list(STUDY_YEARS)
    if years != expected_years:
        missing = sorted(set(expected_years).difference(years))
        raise ValueError(f'The annual source series is incomplete; missing years: {missing}')
    return sources

def source_values(source: rasterio.io.DatasetReader) -> set[int]:
    values: set[int] = set()
    for _, window in source.block_windows(1):
        block = source.read(1, window=window, masked=True)
        values.update((int(value) for value in np.unique(block.compressed())))
    return values

def validate_source(source: rasterio.io.DatasetReader, path: Path) -> set[int]:
    if source.count != 1:
        raise ValueError(f'Expected one band, found {source.count}: {path}')
    if source.crs is None:
        raise ValueError(f'Source raster has no CRS: {path}')
    if source.dtypes[0] != 'uint8':
        raise TypeError(f'Expected uint8 source data, found {source.dtypes[0]}: {path}')
    values = source_values(source)
    if not values:
        raise ValueError(f'Source raster contains no valid pixels: {path}')
    if min(values) < 0 or max(values) >= OUTPUT_NODATA:
        raise ValueError(f'Source classes cannot be represented with NoData={OUTPUT_NODATA}: {path}; values={sorted(values)}')
    return values

def validate_reference(reference: rasterio.io.DatasetReader, path: Path) -> None:
    if reference.count < 1:
        raise ValueError(f'Reference raster has no bands: {path}')
    if reference.crs is None:
        raise ValueError(f'Reference raster has no CRS: {path}')
    valid_count = 0
    for _, window in reference.block_windows(1):
        valid_count += int(np.count_nonzero(reference.read_masks(1, window=window)))
    if valid_count == 0:
        raise ValueError(f'Reference mask has no valid pixels: {path}')

def output_profile(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> dict:
    profile = source.profile.copy()
    profile.update(driver='GTiff', crs=reference.crs, transform=reference.transform, width=reference.width, height=reference.height, count=1, dtype='uint8', nodata=OUTPUT_NODATA, compress='deflate', predictor=1, tiled=True, blockxsize=256, blockysize=256, interleave='band', BIGTIFF='IF_SAFER')
    return profile

def mask_outside_reference(destination: rasterio.io.DatasetWriter, reference: rasterio.io.DatasetReader) -> None:
    for _, window in destination.block_windows(1):
        data = destination.read(1, window=window)
        invalid = reference.read_masks(1, window=window) == 0
        if np.any(invalid):
            data[invalid] = OUTPUT_NODATA
            destination.write(data, 1, window=window)

def validate_output(output_path: Path, reference: rasterio.io.DatasetReader, allowed_values: set[int]) -> None:
    with rasterio.open(output_path) as output:
        mismatches: list[str] = []
        if output.crs != reference.crs:
            mismatches.append('CRS')
        if output.transform != reference.transform:
            mismatches.append('transform')
        if output.width != reference.width or output.height != reference.height:
            mismatches.append('dimensions')
        if output.count != 1:
            mismatches.append('band count')
        if output.dtypes[0] != 'uint8':
            mismatches.append('dtype')
        if output.nodata != OUTPUT_NODATA:
            mismatches.append('NoData')
        if output.overviews(1) != list(OVERVIEW_LEVELS):
            mismatches.append('overviews')
        if mismatches:
            raise RuntimeError(f"Output metadata validation failed ({', '.join(mismatches)}): {output_path}")
        observed_values: set[int] = set()
        outside_valid_pixels = 0
        for _, window in output.block_windows(1):
            data = output.read(1, window=window)
            reference_valid = reference.read_masks(1, window=window) != 0
            outside_valid_pixels += int(np.count_nonzero((data != OUTPUT_NODATA) & ~reference_valid))
            observed_values.update((int(value) for value in np.unique(data[data != OUTPUT_NODATA])))
        unexpected = observed_values.difference(allowed_values)
        if unexpected:
            raise RuntimeError(f'Output contains unexpected class values {sorted(unexpected)}: {output_path}')
        if outside_valid_pixels:
            raise RuntimeError(f'Output has {outside_valid_pixels} valid pixels outside the reference mask: {output_path}')

def align_one(source_path: Path, output_path: Path, reference: rasterio.io.DatasetReader, *, year: int, overwrite: bool, num_threads: int) -> str:
    with rasterio.open(source_path) as source:
        allowed_values = validate_source(source, source_path)
        if output_path.exists() and (not overwrite):
            validate_output(output_path, reference, allowed_values)
            return 'skipped (existing output passed validation)'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output_path.stem}.', suffix='.tmp.tif', dir=output_path.parent)
        os.close(descriptor)
        temporary_path = remap_path(temporary_name)
        try:
            with rasterio.open(temporary_path, 'w+', **output_profile(source, reference)) as destination:
                reproject(source=rasterio.band(source, 1), destination=rasterio.band(destination, 1), src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata, dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=OUTPUT_NODATA, resampling=Resampling.nearest, num_threads=num_threads, warp_mem_limit=512, init_dest_nodata=True)
                mask_outside_reference(destination, reference)
                source_tags = source.tags()
                if source_tags:
                    destination.update_tags(**source_tags)
                destination.update_tags(ALIGNMENT='exact reference grid', REFERENCE_RASTER=str(reference.name), SOURCE_RASTER=str(source_path), RESAMPLING='nearest', STUDY_AREA_MASK='reference valid-data mask', YEAR=str(year), AREA_OR_POINT='Area')
                description = source.descriptions[0] or 'MODIS MCD12C1 LULC class'
                destination.set_band_description(1, description)
                destination.build_overviews(OVERVIEW_LEVELS, Resampling.nearest)
                destination.update_tags(ns='rio_overview', resampling='nearest')
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    validate_output(output_path, reference, allowed_values)
    return 'created and validated'

def main() -> None:
    args = parse_args()
    if args.num_threads < 1:
        raise ValueError('--num-threads must be at least 1')
    if not args.reference.is_file():
        raise FileNotFoundError(f'Reference raster does not exist: {args.reference}')
    sources = collect_sources(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_CACHEMAX=512 * 1024 * 1024):
        with rasterio.open(args.reference) as reference:
            validate_reference(reference, args.reference)
            for index, (year, source_path) in enumerate(sources, start=1):
                output_path = args.output_dir / source_path.with_suffix('.tif').name
                result = align_one(source_path, output_path, reference, year=year, overwrite=args.overwrite, num_threads=args.num_threads)
                print(f'[{index:02d}/{len(sources):02d}] {year}: {result} -> {output_path}')
    print(f'Completed {len(sources)} annual rasters ({sources[0][0]}-{sources[-1][0]}): {args.output_dir}')
if __name__ == '__main__':
    main()
