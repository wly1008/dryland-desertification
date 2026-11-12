"""Align raster inputs to the configured reference grid."""
from __future__ import annotations
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import logging
import math
import os
import tempfile
from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from rasterio.dtypes import check_dtype, in_dtype_range
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject
logger = logging.getLogger(__name__)
_SPATIAL_PROFILE_KEYS = {'crs', 'transform', 'width', 'height', 'count', 'dtype', 'nodata'}

def _open_raster(raster: Any, stack: ExitStack, parameter_name: str):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    if isinstance(raster, (str, PathLike)):
        return stack.enter_context(rasterio.open(raster))
    required_attributes = ('read', 'crs', 'transform', 'width', 'height', 'count', 'dtypes', 'profile')
    if not all((hasattr(raster, name) for name in required_attributes)):
        raise TypeError(f'{parameter_name} must be a raster path or an open Rasterio dataset')
    if getattr(raster, 'closed', False):
        raise ValueError(f'{parameter_name} Rasterio dataset is closed')
    return raster

def _parse_resampling(value: Resampling | str | int) -> Resampling:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    if isinstance(value, Resampling):
        return value
    if isinstance(value, str):
        try:
            return Resampling[value.lower()]
        except KeyError as exc:
            names = ', '.join((item.name for item in Resampling))
            raise ValueError(f'Unknown resampling method {value!r}; available values: {names}') from exc
    if isinstance(value, (int, np.integer)):
        try:
            return Resampling(int(value))
        except ValueError as exc:
            raise ValueError(f'Unknown resampling method identifier: {value}') from exc
    raise TypeError('resampling must be a Resampling enum, method-name string, or integer identifier')

def _normalize_dtype_nodata(source_dtype: str, source_nodata: int | float | None, dtype: str | np.dtype | None, nodata: int | float | None) -> tuple[np.dtype, int | float]:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    output_dtype = np.dtype(source_dtype if dtype is None else dtype)
    if not check_dtype(output_dtype.name):
        raise TypeError(f'Rasterio does not support output data type: {output_dtype}')
    output_nodata = source_nodata if nodata is None else nodata
    if output_nodata is None:
        if np.issubdtype(output_dtype, np.integer):
            output_nodata = np.iinfo(output_dtype).min
        elif np.issubdtype(output_dtype, np.floating):
            output_nodata = np.nan
        else:
            raise TypeError(f'Source raster has no NoData; cannot automatically select NoData for {output_dtype} automatically select NoData')
    if np.issubdtype(output_dtype, np.integer) and isinstance(output_nodata, (float, np.floating)) and (not float(output_nodata).is_integer()):
        raise ValueError(f'NoData={output_nodata!r} is not an integer and cannot be safely written as {output_dtype}')
    if not in_dtype_range(output_nodata, output_dtype.name):
        raise ValueError(f'NoData={output_nodata!r} exceeds the range of data type {output_dtype} representable range')
    with np.errstate(over='raise', invalid='raise'):
        try:
            output_nodata = np.asarray(output_nodata, dtype=output_dtype).item()
        except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f'NoData={output_nodata!r} cannot be safely converted to {output_dtype}') from exc
    return (output_dtype, output_nodata)

def _resolve_mask_bands(reference, data_count: int, mask_band):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    available_bands = tuple(reference.indexes)
    if mask_band is None:
        selected_bands = available_bands
    elif isinstance(mask_band, (int, np.integer)) and (not isinstance(mask_band, (bool, np.bool_))):
        selected_band = int(mask_band)
        if selected_band not in available_bands:
            raise IndexError(f'mask_band={selected_band} exceeds reference band range {available_bands}')
        return selected_band
    elif isinstance(mask_band, tuple):
        if not mask_band:
            raise ValueError('mask_band tuple must not be empty')
        if not all((isinstance(item, (int, np.integer)) and (not isinstance(item, (bool, np.bool_))) for item in mask_band)):
            raise TypeError('mask_band tuple must contain integer band numbers only')
        selected_bands = tuple((int(item) for item in mask_band))
    else:
        raise TypeError('mask_band must be None, an integer, or an integer tuple')
    if len(set(selected_bands)) != len(selected_bands):
        raise ValueError('mask_band groups must not contain duplicate band numbers')
    invalid_bands = tuple((band for band in selected_bands if band not in available_bands))
    if invalid_bands:
        raise IndexError(f'mask_band group bands {invalid_bands} exceeds reference band range {available_bands}')
    if len(selected_bands) != data_count:
        selection_label = 'All reference-raster bands' if mask_band is None else 'mask_band group'
        raise ValueError(f'{selection_label}count is {len(selected_bands)}; data have {data_count} bands. Group lengths must match; use an integer band number to apply one mask band to all data bands.')
    return selected_bands

def _iter_fixed_windows(width: int, height: int, data_count: int, data_dtype: str | np.dtype, mask_count: int, mem_limit: int | float):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    bytes_per_pixel = data_count * np.dtype(data_dtype).itemsize + 2 * mask_count
    usable_bytes = max(1, int(float(mem_limit) * 1024 ** 2 * 0.8))
    max_pixels = max(1, usable_bytes // bytes_per_pixel)
    square_side = max(1, math.isqrt(max_pixels))
    window_width = min(width, square_side)
    window_height = min(height, max(1, max_pixels // window_width))
    row_windows = math.ceil(height / window_height)
    column_windows = math.ceil(width / window_width)
    logger.info('Fixed mask windows: %d columns by %d rows; count=%d (mem_limit=%g MiB)', window_width, window_height, row_windows * column_windows, float(mem_limit))
    for row_off in range(0, height, window_height):
        current_height = min(window_height, height - row_off)
        for col_off in range(0, width, window_width):
            current_width = min(window_width, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=current_width, height=current_height)

def _read_invalid_mask(reference, window: Window, mask_bands):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    return reference.read_masks(mask_bands, window=window) == 0

def _apply_invalid_mask(block: np.ndarray, invalid: np.ndarray, nodata) -> None:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    if invalid.ndim == 2:
        block[:, invalid] = nodata
    else:
        block[invalid] = nodata

def _apply_reference_mask_to_array(array: np.ndarray, reference, nodata: int | float, mask_bands, mem_limit: int | float) -> None:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    mask_count = 1 if isinstance(mask_bands, int) else len(mask_bands)
    windows = _iter_fixed_windows(width=reference.width, height=reference.height, data_count=array.shape[0], data_dtype=array.dtype, mask_count=mask_count, mem_limit=mem_limit)
    for window in windows:
        invalid = _read_invalid_mask(reference, window, mask_bands)
        if not np.any(invalid):
            continue
        row_slice, col_slice = window.toslices()
        block = array[:, row_slice, col_slice]
        _apply_invalid_mask(block, invalid, nodata)

def _apply_reference_mask_to_dataset(dataset, reference, nodata, mask_bands, mem_limit: int | float) -> None:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    mask_count = 1 if isinstance(mask_bands, int) else len(mask_bands)
    windows = _iter_fixed_windows(width=dataset.width, height=dataset.height, data_count=dataset.count, data_dtype=dataset.dtypes[0], mask_count=mask_count, mem_limit=mem_limit)
    for window in windows:
        invalid = _read_invalid_mask(reference, window, mask_bands)
        if not np.any(invalid):
            continue
        block = dataset.read(window=window)
        _apply_invalid_mask(block, invalid, nodata)
        dataset.write(block, window=window)

def _copy_nonspatial_metadata(source, destination) -> None:
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    dataset_tags = source.tags()
    if dataset_tags:
        destination.update_tags(**dataset_tags)
    for band_index in range(1, source.count + 1):
        description = source.descriptions[band_index - 1]
        if description is not None:
            destination.set_band_description(band_index, description)
        band_tags = source.tags(band_index)
        if band_tags:
            destination.update_tags(band_index, **band_tags)

def align_raster(source, reference, out_path: str | PathLike | None=None, *, nodata: int | float | None=None, dtype: str | np.dtype | None=None, resampling: Resampling | str | int=Resampling.nearest, apply_mask: bool=False, mask_band: int | tuple[int, ...] | None=None, mem_limit: int | float=512, overwrite: bool=False, num_threads: int=4, **profile_updates):
    """Implementation notes are summarized in docs/ALGORITHM_NOTES.md."""
    if not isinstance(apply_mask, (bool, np.bool_)):
        raise TypeError('apply_mask must be boolean')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError('overwrite must be boolean')
    if not isinstance(mem_limit, (int, float, np.integer, np.floating)):
        raise TypeError('mem_limit must be positive and expressed in MiB')
    if not np.isfinite(mem_limit) or mem_limit <= 0:
        raise ValueError('mem_limit must be positive and expressed in MiB')
    if not isinstance(num_threads, (int, np.integer)) or num_threads < 1:
        raise ValueError('num_threads must be an integer >=1')
    conflicts = _SPATIAL_PROFILE_KEYS.intersection(profile_updates)
    if conflicts:
        raise ValueError('These attributes are controlled internally and cannot be overwritten through profile_updates: ' + ', '.join(sorted(conflicts)))
    resampling_method = _parse_resampling(resampling)
    warp_mem_limit = max(1, int(mem_limit))
    with ExitStack() as stack:
        src = _open_raster(source, stack, 'source')
        ref = _open_raster(reference, stack, 'reference')
        if src.crs is None:
            raise ValueError('source has no CRS and cannot be aligned safely')
        if ref.crs is None:
            raise ValueError('reference has no CRS and cannot be aligned safely')
        if src.count < 1:
            raise ValueError('source has no processable raster bands')
        if ref.count < 1:
            raise ValueError('reference has no usable reference bands')
        selected_mask_bands = None
        if apply_mask:
            selected_mask_bands = _resolve_mask_bands(reference=ref, data_count=src.count, mask_band=mask_band)
        output_dtype, output_nodata = _normalize_dtype_nodata(source_dtype=src.dtypes[0], source_nodata=src.nodata, dtype=dtype, nodata=nodata)
        output_profile = src.profile.copy()
        output_profile.update(driver='GTiff', crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height, count=src.count, dtype=output_dtype.name, nodata=output_nodata)
        output_profile.update(profile_updates)
        logger.info('Aligned raster: %d bands; target %d rows by %d columns; resampling=%s', src.count, ref.height, ref.width, resampling_method.name)
        reproject_options = dict(src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata, dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=output_nodata, resampling=resampling_method, num_threads=int(num_threads), warp_mem_limit=warp_mem_limit, init_dest_nodata=True)
        if out_path is None:
            output_bytes = src.count * ref.height * ref.width * output_dtype.itemsize
            limit_bytes = int(float(mem_limit) * 1024 * 1024)
            if output_bytes > limit_bytes:
                raise MemoryError(f'Returned array would occupy {output_bytes / 1024 ** 2:.2f} MiB, exceeding mem_limit={float(mem_limit):g} MiB; provide out_path for streamed disk output')
            destination = np.full((src.count, ref.height, ref.width), output_nodata, dtype=output_dtype)
            for band_index in range(1, src.count + 1):
                reproject(source=rasterio.band(src, band_index), destination=destination[band_index - 1], **reproject_options)
            if apply_mask:
                _apply_reference_mask_to_array(destination, ref, output_nodata, selected_mask_bands, mem_limit)
            return (destination, output_profile)
        output_path = remap_path(out_path)
        if output_path.exists() and (not overwrite):
            raise FileExistsError(f'Output exists and overwrite=False: {output_path}')
        for input_value, input_name in ((source, 'source'), (reference, 'reference')):
            if isinstance(input_value, (str, PathLike)):
                if output_path.resolve() == remap_path(input_value).resolve():
                    raise ValueError(f'out_path must differ from {input_name} path')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output_path.stem}.', suffix=f".tmp{output_path.suffix or '.tif'}", dir=output_path.parent)
        os.close(file_descriptor)
        temporary_path = remap_path(temporary_name)
        try:
            with rasterio.open(temporary_path, 'w+', **output_profile) as dst:
                for band_index in range(1, src.count + 1):
                    logger.info('Reproject/resample band %d/%d', band_index, src.count)
                    reproject(source=rasterio.band(src, band_index), destination=rasterio.band(dst, band_index), **reproject_options)
                if apply_mask:
                    logger.info('Apply the valid reference-raster mask')
                    _apply_reference_mask_to_dataset(dst, ref, output_nodata, selected_mask_bands, mem_limit)
                _copy_nonspatial_metadata(src, dst)
            if output_path.exists() and (not overwrite):
                raise FileExistsError(f'Output was created during processing and overwrite=False: {output_path}')
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        logger.info('Saved: %s', output_path)
        return str(output_path)
