"""Calculate raster percentile statistics for normalization."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, VRT_ROOT, raster_source
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
index = 'Albedo'
raster_path = raster_source(VRT_ROOT / 'rs_indices' / index / '2003', f'{index}_2003_04')
band = 1
nodata = None
valid_min = None
valid_max = None
block = 4096
nbins = 200000
range_min = None
range_max = None
percentiles = [0, 0.1, 1, 2, 50, 98, 99, 99.9, 100]
query_values = [-100, -50, -20, 0, 20, 50, 100]
query_values = [i * 100 for i in query_values]
hist_csv = None
plot_hist = True
plot_xlim = (-100, 100)
plot_xlim = [i * 100 for i in plot_xlim]
plot_ylim = None
plot_bins = 500
plot_log_y = False
plot_save = None
src = rasterio.open(raster_path)
print(f'Raster: {raster_path}')
print(f'Dimensions: {src.width} x {src.height}; band {band}; data type {src.dtypes[band - 1]}')
if nodata is None and src.nodata is not None:
    nodata = src.nodata
    print(f'Use file NoData metadata = {nodata}')
n_rows = (src.height + block - 1) // block
n_cols = (src.width + block - 1) // block
total_blocks = n_rows * n_cols
if range_min is None or range_max is None:
    vmin, vmax = (np.inf, -np.inf)
    done = 0
    for row_off in range(0, src.height, block):
        for col_off in range(0, src.width, block):
            h = min(block, src.height - row_off)
            w = min(block, src.width - col_off)
            arr = src.read(band, window=Window(col_off, row_off, w, h)).ravel()
            mask = np.isfinite(arr)
            if nodata is not None:
                mask &= arr != nodata
            if valid_min is not None:
                mask &= arr >= valid_min
            if valid_max is not None:
                mask &= arr <= valid_max
            arr = arr[mask]
            if arr.size:
                vmin = min(vmin, float(arr.min()))
                vmax = max(vmax, float(arr.max()))
            done += 1
            print(f'\rPass one (range): {done}/{total_blocks} blocks', end='', flush=True)
    print()
    if not np.isfinite(vmin):
        raise ValueError('No valid pixels; check NoData and valid-value limits')
    if range_min is None:
        range_min = vmin
    if range_max is None:
        range_max = vmax
print(f'Bin range: [{range_min}, {range_max}]; bins: {nbins}; bin width (percentile resolution): {(range_max - range_min) / nbins:.6g}')
edges = np.linspace(range_min, range_max, nbins + 1)
counts = np.zeros(nbins, dtype=np.int64)
done = 0
for row_off in range(0, src.height, block):
    for col_off in range(0, src.width, block):
        h = min(block, src.height - row_off)
        w = min(block, src.width - col_off)
        arr = src.read(band, window=Window(col_off, row_off, w, h)).ravel()
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        mask &= (arr >= range_min) & (arr <= range_max)
        arr = arr[mask]
        if arr.size:
            c, _ = np.histogram(arr, bins=nbins, range=(range_min, range_max))
            counts += c
        done += 1
        print(f'\rPass two (histogram): {done}/{total_blocks} blocks', end='', flush=True)
src.close()
print()
centers = (edges[:-1] + edges[1:]) / 2
cum = np.cumsum(counts)
total = int(cum[-1])
print(f'Valid pixels: {total:,}')
print(f'Minimum: {range_min:.6g}; maximum: {range_max:.6g}')
print(f'Mean (approximated by bin centers): {(centers * counts).sum() / total:.6g}')
if hist_csv:
    np.savetxt(hist_csv, np.column_stack([centers, counts]), fmt='%.8g,%d', header='bin_center,count', comments='')
    print(f'Histogram saved: {hist_csv}')
print('\nPercentile values')
bin_width = (range_max - range_min) / nbins
for p in percentiles:
    target = p / 100.0 * total
    idx = int(np.searchsorted(cum, target, side='left'))
    idx = min(idx, nbins - 1)
    prev = cum[idx - 1] if idx > 0 else 0
    frac = (target - prev) / counts[idx] if counts[idx] > 0 else 0.5
    val = edges[idx] + frac * bin_width
    print(f'  P{p:g} = {val:.6g}')
print('\nRanks of specified values')
for v in query_values:
    if v < range_min:
        print(f'  value {v:g}: 0.0000% of pixels are <= this value (below minimum)')
        continue
    if v >= range_max:
        print(f'  value {v:g}: 100.0000% of pixels are <= this value (at or above maximum)')
        continue
    idx = int((v - range_min) / bin_width)
    idx = min(idx, nbins - 1)
    prev = cum[idx - 1] if idx > 0 else 0
    frac = (v - edges[idx]) / bin_width
    n_le = prev + frac * counts[idx]
    print(f'  value {v:g}: {100.0 * n_le / total:.4f}% of pixels are <= this value')
if plot_hist:
    factor = max(1, nbins // plot_bins)
    n_plot = nbins // factor
    plot_counts = counts[:n_plot * factor].reshape(n_plot, factor).sum(axis=1)
    plot_edges = edges[::factor][:n_plot + 1]
    plot_centers = (plot_edges[:-1] + plot_edges[1:]) / 2
    plot_width = plot_edges[1] - plot_edges[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_centers, plot_counts, width=plot_width, color='steelblue', edgecolor='none')
    if plot_xlim is not None:
        ax.set_xlim(plot_xlim)
    if plot_ylim is not None:
        ax.set_ylim(plot_ylim)
    if plot_log_y:
        ax.set_yscale('log')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Pixel count')
    ax.set_title(f'{index}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    if plot_save:
        fig.savefig(figure_output_path(plot_save), dpi=200)
        print(f'\nHistogram image saved: {plot_save}')
    plt.show()
