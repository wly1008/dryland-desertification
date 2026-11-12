"""Reclassify MODIS land cover to six classes and align it to the reference grid."""
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / '02_preprocessing/data/auxiliary/land_cover_0p1/class'
SOURCE_PATTERN = 'MCD12C1_LULC_*.tif'
YEAR_PATTERN = re.compile('^MCD12C1_LULC_(\\d{4})\\.(?:tif|tiff|vrt)$')
OUTPUT_NAME = 'MCD12C1_LULC_CLASS_{year}.tif'
STUDY_YEARS = tuple(range(2001, 2022))
OUTPUT_NODATA = 255
OVERVIEW_LEVELS = (2, 4, 8, 16, 32)
CLASS_DEFINITIONS = {1: ('Forest Land', (1, 2, 3, 4, 5)), 2: ('Grassland', (6, 7, 8, 9, 10)), 3: ('Cropland', (12, 14)), 4: ('Wetlands', (11,)), 5: ('Settlements', (13,)), 6: ('Other Land', (0, 15, 16))}
RECLASS_LOOKUP = np.full(256, OUTPUT_NODATA, dtype='uint8')
for output_code, (_, source_codes) in CLASS_DEFINITIONS.items():
    RECLASS_LOOKUP[list(source_codes)] = output_code
EXPECTED_SOURCE_CODES = frozenset(range(17))
EXPECTED_OUTPUT_CODES = frozenset(CLASS_DEFINITIONS)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reclassify annual MODIS MCD12C1 IGBP rasters to six classes, then align them to the project's 0.1-degree arid-region grid.")
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing outputs. Otherwise, valid existing outputs are skipped.')
    parser.add_argument('--num-threads', type=int, default=4)
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
        raise FileNotFoundError(f'No files matching {SOURCE_PATTERN!r} were found in {source_dir}')
    years = [year for year, _ in sources]
    expected_years = list(STUDY_YEARS)
    if len(years) != len(set(years)):
        raise ValueError(f'Duplicate years found: {years}')
    if years != expected_years:
        missing = sorted(set(expected_years).difference(years))
        raise ValueError(f'The annual source series is incomplete; missing: {missing}')
    return sources

def validate_reference(reference: rasterio.io.DatasetReader, path: Path) -> None:
    if reference.count < 1:
        raise ValueError(f'Reference raster has no bands: {path}')
    if reference.crs is None:
        raise ValueError(f'Reference raster has no CRS: {path}')
    if not np.any(reference.read_masks(1)):
        raise ValueError(f'Reference raster has no valid pixels: {path}')

def read_and_reclassify(source: rasterio.io.DatasetReader, source_path: Path) -> tuple[np.ndarray, set[int]]:
    if source.count != 1:
        raise ValueError(f'Expected one source band, found {source.count}: {source_path}')
    if source.crs is None:
        raise ValueError(f'Source raster has no CRS: {source_path}')
    if source.dtypes[0] != 'uint8':
        raise TypeError(f'Expected uint8 source, found {source.dtypes[0]}: {source_path}')
    source_data = source.read(1)
    observed_source_codes = set((int(value) for value in np.unique(source_data)))
    unexpected = observed_source_codes.difference(EXPECTED_SOURCE_CODES)
    if unexpected:
        raise ValueError(f'Unmapped source codes {sorted(unexpected)} in {source_path}')
    reclassified = RECLASS_LOOKUP[source_data]
    observed_output_codes = set((int(value) for value in np.unique(reclassified)))
    if OUTPUT_NODATA in observed_output_codes:
        raise RuntimeError(f'Reclassification produced unmapped pixels: {source_path}')
    return (reclassified, observed_output_codes)

def build_output_profile(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> dict:
    profile = source.profile.copy()
    profile.update(driver='GTiff', crs=reference.crs, transform=reference.transform, width=reference.width, height=reference.height, count=1, dtype='uint8', nodata=OUTPUT_NODATA, compress='deflate', predictor=1, tiled=True, blockxsize=256, blockysize=256, interleave='band', BIGTIFF='IF_SAFER')
    return profile

def apply_reference_mask(destination: rasterio.io.DatasetWriter, reference: rasterio.io.DatasetReader) -> None:
    for _, window in destination.block_windows(1):
        data = destination.read(1, window=window)
        invalid = reference.read_masks(1, window=window) == 0
        if np.any(invalid):
            data[invalid] = OUTPUT_NODATA
            destination.write(data, 1, window=window)

def validate_output(output_path: Path, reference: rasterio.io.DatasetReader) -> dict[int, int]:
    class_counts = {code: 0 for code in sorted(EXPECTED_OUTPUT_CODES)}
    outside_valid = 0
    missing_inside = 0
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
        observed_codes: set[int] = set()
        for _, window in output.block_windows(1):
            data = output.read(1, window=window)
            reference_valid = reference.read_masks(1, window=window) != 0
            outside_valid += int(np.count_nonzero((data != OUTPUT_NODATA) & ~reference_valid))
            missing_inside += int(np.count_nonzero((data == OUTPUT_NODATA) & reference_valid))
            valid_data = data[data != OUTPUT_NODATA]
            observed_codes.update((int(value) for value in np.unique(valid_data)))
            for code in class_counts:
                class_counts[code] += int(np.count_nonzero(valid_data == code))
    unexpected = observed_codes.difference(EXPECTED_OUTPUT_CODES)
    if unexpected:
        raise RuntimeError(f'Output contains unexpected class codes {sorted(unexpected)}: {output_path}')
    if outside_valid:
        raise RuntimeError(f'Found {outside_valid} valid pixels outside reference mask: {output_path}')
    if missing_inside:
        raise RuntimeError(f'Found {missing_inside} NoData pixels inside reference mask: {output_path}')
    return class_counts

def align_one(source_path: Path, output_path: Path, reference: rasterio.io.DatasetReader, *, year: int, overwrite: bool, num_threads: int) -> tuple[str, dict[int, int]]:
    if output_path.exists() and (not overwrite):
        counts = validate_output(output_path, reference)
        return ('skipped (existing output passed validation)', counts)
    with rasterio.open(source_path) as source:
        reclassified, source_output_codes = read_and_reclassify(source, source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output_path.stem}.', suffix='.tmp.tif', dir=output_path.parent)
        os.close(descriptor)
        temporary_path = remap_path(temporary_name)
        try:
            with rasterio.open(temporary_path, 'w+', **build_output_profile(source, reference)) as destination:
                reproject(source=reclassified, destination=rasterio.band(destination, 1), src_transform=source.transform, src_crs=source.crs, src_nodata=None, dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=OUTPUT_NODATA, resampling=Resampling.nearest, num_threads=num_threads, warp_mem_limit=512, init_dest_nodata=True)
                apply_reference_mask(destination, reference)
                source_tags = source.tags()
                if source_tags:
                    destination.update_tags(**source_tags)
                destination.update_tags(ALIGNMENT='exact reference grid', PROCESSING_ORDER='reclassify then align', REFERENCE_RASTER=str(reference.name), SOURCE_RASTER=str(source_path), RESAMPLING='nearest', STUDY_AREA_MASK='reference valid-data mask', YEAR=str(year), CLASS_1='Forest Land: source 1,2,3,4,5', CLASS_2='Grassland: source 6,7,8,9,10', CLASS_3='Cropland: source 12,14', CLASS_4='Wetlands: source 11', CLASS_5='Settlements: source 13', CLASS_6='Other Land: source 0,15,16', AREA_OR_POINT='Area')
                destination.set_band_description(1, 'Reclassified MODIS MCD12C1 LULC (six classes)')
                destination.build_overviews(OVERVIEW_LEVELS, Resampling.nearest)
                destination.update_tags(ns='rio_overview', resampling='nearest')
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    counts = validate_output(output_path, reference)
    observed_output_codes = {code for code, count in counts.items() if count > 0}
    if not observed_output_codes.issubset(source_output_codes):
        raise RuntimeError(f'Output validation inconsistency: {output_path}')
    return ('created and validated', counts)

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
                output_path = args.output_dir / OUTPUT_NAME.format(year=year)
                status, counts = align_one(source_path, output_path, reference, year=year, overwrite=args.overwrite, num_threads=args.num_threads)
                count_text = ', '.join((f'{code}:{counts[code]}' for code in sorted(counts)))
                print(f'[{index:02d}/{len(sources):02d}] {year}: {status}; class counts [{count_text}] -> {output_path}')
    print(f'Completed {len(sources)} annual rasters ({sources[0][0]}-{sources[-1][0]}): {args.output_dir}')
if __name__ == '__main__':
    main()
