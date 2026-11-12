"""Summarize DSI responses within cover-change combinations."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
import json
import csv
import numpy as np
import rasterio
from tqdm import tqdm
ROOT = PROJECT_ROOT / '03_analysis/05_cover_combinations'
OUT = PROJECT_ROOT / '03_analysis/05_cover_combinations/data'
DSI_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
DSI_SIGNIFICANCE_ROOT = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
DSI_TREND_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_trend_2001_2021.tif'
DSI_PVALUE_PATH = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights/DI_significance_pvalue_2001_2021.tif'
PRIMARY_ALPHA = 0.05
TOP_N = 6
REP_LEVELS = [(0.7, 100), (0.5, 50), (0.3, 25), (0.2, 10)]
base = json.loads((PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combination_summary.json').read_text(encoding='utf-8'))
pixels = np.load(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_pixel_statistics.npz')
cover = np.load(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_annual_cover_pp.npy', mmap_mode='r')
rows, cols = (pixels['rows'], pixels['cols'])
area, labels = (pixels['area_km2'], pixels['combination'])
trends, trend_pvalue = (pixels['trend_pp_per_year'], pixels['trend_pvalue'])
years = pixels['years']
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combination_2001_2021.tif') as src:
    profile, shape, transform, crs = (src.profile.copy(), src.shape, src.transform, src.crs)
dsi_all = np.empty((len(rows), len(years)), np.float32)
paired = np.ones(len(rows), bool)
sources = []
for t, year in enumerate(tqdm(years, desc='Read main-product DSI', unit='year')):
    path = DSI_ROOT / f'DI_{year}.tif'
    with rasterio.open(path) as src:
        if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
            raise ValueError(f'DSI grid mismatch: {path}')
        a = src.read(1, masked=True)
        paired &= (~np.ma.getmaskarray(a) & np.isfinite(a.data))[rows, cols]
        dsi_all[:, t] = a.data[rows, cols]
    sources.append({'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
with rasterio.open(DSI_TREND_PATH) as src:
    if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
        raise ValueError(f'DSI trend grid mismatch: {DSI_TREND_PATH}')
    a = src.read(1, masked=True)
    dsi_trend_all = a.data[rows, cols].astype(np.float64)
    paired &= (~np.ma.getmaskarray(a) & np.isfinite(a.data))[rows, cols]
with rasterio.open(DSI_PVALUE_PATH) as src:
    if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
        raise ValueError(f'DSI p-value grid mismatch: {DSI_PVALUE_PATH}')
    a = src.read(1, masked=True)
    dsi_pvalue_all = a.data[rows, cols].astype(np.float64)
    paired &= (~np.ma.getmaskarray(a) & np.isfinite(a.data))[rows, cols]
sources.extend([{'path': DSI_TREND_PATH.relative_to(PROJECT_ROOT).as_posix(), 'size': DSI_TREND_PATH.stat().st_size, 'mtime_ns': DSI_TREND_PATH.stat().st_mtime_ns}, {'path': DSI_PVALUE_PATH.relative_to(PROJECT_ROOT).as_posix(), 'size': DSI_PVALUE_PATH.stat().st_size, 'mtime_ns': DSI_PVALUE_PATH.stat().st_mtime_ns}])
paired_indices = np.flatnonzero(paired)
cover = np.asarray(cover[paired])
dsi = dsi_all[paired].astype(np.float64)
del dsi_all
rows, cols = (rows[paired], cols[paired])
area, labels = (area[paired], labels[paired])
trends, trend_pvalue = (trends[paired], trend_pvalue[paired])
dsi_trend, dsi_pvalue = (dsi_trend_all[paired], dsi_pvalue_all[paired])
time = years - years.mean()
annual_slope_check = dsi @ time / np.sum(time ** 2)
trend_check_max_abs_difference = float(np.max(np.abs(annual_slope_check - dsi_trend)))
dsi_state = np.ones(len(dsi_trend), np.uint8)
dsi_significant = dsi_pvalue < PRIMARY_ALPHA
dsi_state[dsi_significant & (dsi_trend < 0)] = 0
dsi_state[dsi_significant & (dsi_trend > 0)] = 2
area_by_class = np.bincount(labels, weights=area, minlength=28)[1:]
full_area = np.asarray([r['area_percent'] for r in base['combinations']])
top_ids = (np.argsort(-full_area)[:TOP_N] + 1).tolist()
class_records = []
fvc_series = np.full((27, len(years), 3), np.nan)
dsi_series = np.full((27, len(years)), np.nan)
dsi_distributions = np.full((27, 5), np.nan)
fvc_distributions = np.full((27, 3, 3), np.nan)
quantile_levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
for g in range(1, 28):
    selected = labels == g
    rec = dict(base['combinations'][g - 1])
    rec['paired_pixels'] = int(selected.sum())
    rec['paired_area_percent'] = float(area_by_class[g - 1] / area.sum() * 100)
    full_class_area = rec['area_percent'] / 100 * base['area_km2']
    rec['area_retained_percent'] = float(area_by_class[g - 1] / full_class_area * 100) if full_class_area else None
    if selected.any():
        w = area[selected]
        fvc_series[g - 1] = np.average(cover[selected], weights=w, axis=0)
        dsi_series[g - 1] = np.average(dsi[selected], weights=w, axis=0)
        vals = dsi_trend[selected]
        order = np.argsort(vals)
        midpoint = (np.cumsum(w[order]) - w[order] / 2) / w.sum()
        dsi_distributions[g - 1] = np.interp(quantile_levels, midpoint, vals[order])
        direction_area = [float(w[dsi_state[selected] == state].sum() / w.sum() * 100) for state in range(3)]
        slope_direction_area = [float(w[vals < 0].sum() / w.sum() * 100), float(w[vals > 0].sum() / w.sum() * 100)]
        rec.update({'dsi_trend_quantiles_per_year': dsi_distributions[g - 1].tolist(), 'dsi_trend_mean_per_year': float(np.average(vals, weights=w)), 'dsi_significance_percent_decrease_NS_increase': direction_area, 'dsi_direction_percent_decrease_increase': slope_direction_area})
        for c in range(3):
            v = trends[selected, c]
            order = np.argsort(v)
            midpoint = (np.cumsum(w[order]) - w[order] / 2) / w.sum()
            fvc_distributions[g - 1, c] = np.interp([0.25, 0.5, 0.75], midpoint, v[order])
        rec['fvc_trend_quartiles_pp_per_year'] = fvc_distributions[g - 1].tolist()
    class_records.append(rec)
coordinate_max = np.max(np.abs(trends[:, :2]), axis=1)
order = np.argsort(coordinate_max)
midpoint = (np.cumsum(area[order]) - area[order] / 2) / area.sum()
limit = max(0.4, float(np.ceil(np.interp(0.995, midpoint, coordinate_max[order]) / 0.2) * 0.2))
edges = np.linspace(-limit, limit, 41)
inside = np.all((trends[:, :2] >= -limit) & (trends[:, :2] <= limit), axis=1)
ix = np.clip(np.searchsorted(edges, trends[inside, 0], side='right') - 1, 0, 39)
iy = np.clip(np.searchsorted(edges, trends[inside, 1], side='right') - 1, 0, 39)
index = iy * 40 + ix
bin_area = np.bincount(index, weights=area[inside], minlength=1600)
bin_bs_sum = np.bincount(index, weights=area[inside] * trends[inside, 2], minlength=1600)
bin_bs = np.divide(bin_bs_sum, bin_area, out=np.full(1600, np.nan), where=bin_area > 0)
bin_count = np.bincount(index, minlength=1600)
states = np.asarray([r['states'] for r in base['combinations']])
example_ids = []
for kind in range(4):
    if kind == 0:
        allowed = (states[:, 0] == 2) & (states[:, 1] == 0)
    elif kind == 1:
        allowed = (states[:, 0] == 0) & (states[:, 1] == 2)
    elif kind == 2:
        allowed = (states[:, 2] == 0) & np.all(states[:, :2] >= 1, axis=1) & np.any(states[:, :2] == 2, axis=1)
    else:
        allowed = np.all(states == 1, axis=1)
    candidates = np.flatnonzero(allowed & (area_by_class > 0))
    if len(candidates):
        example_ids.append(int(candidates[np.argmax(full_area[candidates])] + 1))
for g in example_ids[:2]:
    if g not in top_ids:
        top_ids.append(g)
lon, lat = (transform.c + (cols + 0.5) * transform.a, transform.f + (rows + 0.5) * transform.e)
tile_x, tile_y = (np.floor((lon + 180) / 2).astype(int), np.floor((lat + 90) / 2).astype(int))
tiles, tile_index = np.unique(tile_y * 181 + tile_x, return_inverse=True)
tile_total_area = np.bincount(tile_index, weights=area)
examples = []
region_cover = np.full((len(example_ids), len(years), 3), np.nan)
region_dsi = np.full((len(example_ids), len(years)), np.nan)
for e, g in enumerate(example_ids):
    sel = labels == g
    counts = np.bincount(tile_index[sel], minlength=len(tiles))
    group_area = np.bincount(tile_index[sel], weights=area[sel], minlength=len(tiles))
    purity = group_area / tile_total_area
    distance = np.mean((cover[sel] - fvc_series[g - 1]) ** 2, axis=(1, 2))
    distance_by_tile = np.divide(np.bincount(tile_index[sel], weights=area[sel] * distance, minlength=len(tiles)), group_area, out=np.full(len(tiles), np.inf), where=group_area > 0)
    possible = np.array([], dtype=int)
    level = None
    for purity_min, count_min in REP_LEVELS:
        possible = np.flatnonzero((purity >= purity_min) & (counts >= count_min))
        if len(possible):
            level = [purity_min, count_min]
            break
    if not len(possible):
        examples.append({'id': g, 'available': False, 'reason': 'No region meets predefined 2-degree criteria'})
        continue
    chosen = int(possible[np.argmin(distance_by_tile[possible])])
    members = sel & (tile_index == chosen)
    first = np.flatnonzero(members)[0]
    region_cover[e] = np.average(cover[members], weights=area[members], axis=0)
    region_dsi[e] = np.average(dsi[members], weights=area[members], axis=0)
    mean_trend = time @ region_cover[e] / np.sum(time ** 2)
    examples.append({'id': g, 'available': True, 'lon_min': float(-180 + 2 * tile_x[first]), 'lon_max': float(-180 + 2 * (tile_x[first] + 1)), 'lat_min': float(-90 + 2 * tile_y[first]), 'lat_max': float(-90 + 2 * (tile_y[first] + 1)), 'pixels': int(members.sum()), 'purity': float(purity[chosen]), 'selection_level': level, 'mean_squared_cover_distance': float(distance_by_tile[chosen]), 'cover_trend_pp_per_year': mean_trend.tolist(), 'dsi_trend_per_year': float(time @ region_dsi[e] / np.sum(time ** 2))})
grid = np.full(shape, -9999, np.float32)
grid[rows, cols] = dsi_trend
profile.update(dtype='float32', nodata=-9999, count=1, compress='deflate')
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_DSI_trend_per_year.tif', 'w', **profile) as dst:
    dst.write(grid, 1)
    dst.update_tags(UNIT='DSI per year', PRODUCT='with NPV only', SOURCE=DSI_TREND_PATH.relative_to(PROJECT_ROOT).as_posix())
grid = np.full(shape, -9999, np.float32)
grid[rows, cols] = dsi_pvalue
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_DSI_trend_pvalue.tif', 'w', **profile) as dst:
    dst.write(grid, 1)
    dst.update_tags(UNIT='probability', SOURCE=DSI_PVALUE_PATH.relative_to(PROJECT_ROOT).as_posix())
grid = np.full(shape, 255, np.uint8)
grid[rows, cols] = labels
profile.update(dtype='uint8', nodata=255)
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/paired_fvc_combination.tif', 'w', **profile) as dst:
    dst.write(grid, 1)
np.savez_compressed(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_response_series.npz', years=years, class_fvc_pp=fvc_series, class_dsi=dsi_series, dsi_trend_quantiles_per_year=dsi_distributions, fvc_trend_quartiles_pp_per_year=fvc_distributions, top_ids=np.asarray(top_ids), example_ids=np.asarray(example_ids), region_fvc_pp=region_cover, region_dsi=region_dsi, plane_edges_pp_per_year=edges, plane_bs_trend_pp_per_year=bin_bs.reshape(40, 40), plane_area_km2=bin_area.reshape(40, 40), plane_count=bin_count.reshape(40, 40))
np.savez_compressed(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/paired_pixel_statistics.npz', rows=rows, cols=cols, area_km2=area, combination=labels, fvc_trend_pp_per_year=trends, fvc_trend_pvalue=trend_pvalue, main_dsi_trend_per_year=dsi_trend, main_dsi_trend_pvalue=dsi_pvalue, main_dsi_significance_state=dsi_state, fvc_input_indices=paired_indices)
summary = {'study_path': base['study_path'], 'years': years.tolist(), 'main_dsi_root': DSI_ROOT.relative_to(PROJECT_ROOT).as_posix(), 'dsi_trend_source': DSI_TREND_PATH.relative_to(PROJECT_ROOT).as_posix(), 'dsi_pvalue_source': DSI_PVALUE_PATH.relative_to(PROJECT_ROOT).as_posix(), 'dsi_alpha': PRIMARY_ALPHA, 'full_fvc_pixels': base['complete_fvc_pixels'], 'paired_pixels': len(rows), 'paired_area_km2': float(area.sum()), 'area_retained_percent': float(area.sum() / base['area_km2'] * 100), 'fvc_primary_alpha': base['primary_alpha'], 'quantiles': quantile_levels.tolist(), 'classes': class_records, 'top_ids': top_ids, 'top_selection': 'six largest significance-defined combinations plus the largest combination in each opposing PV/NPV direction, selected by independent FVC-domain area; no DSI-based selection', 'example_ids': example_ids, 'examples': examples, 'plane_limit_pp_per_year': limit, 'plane_retained_area_percent': float(area[inside].sum() / area.sum() * 100), 'DSI_direction_rule': 'Figure 10d uses slope sign only: decrease if slope<0 and increase if slope>0; DSI significance remains available separately', 'dsi_annual_slope_vs_reference_max_abs_difference': trend_check_max_abs_difference, 'sources': sources}
(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_dsi_response_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
with (PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_dsi_response_all27.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['combination', 'PV_state', 'NPV_state', 'BS_state', 'full_FVC_area_percent', 'paired_area_percent', 'paired_pixels', 'area_retained_percent', 'DSI_q05_per_year', 'DSI_q25_per_year', 'DSI_q50_per_year', 'DSI_q75_per_year', 'DSI_q95_per_year', 'DSI_significant_decrease_area_percent', 'DSI_NS_area_percent', 'DSI_significant_increase_area_percent', 'DSI_decrease_direction_area_percent', 'DSI_increase_direction_area_percent'])
    for rec in class_records:
        writer.writerow([rec['id'], *rec['states'], rec['area_percent'], rec['paired_area_percent'], rec['paired_pixels'], rec['area_retained_percent'], *rec.get('dsi_trend_quantiles_per_year', [np.nan] * 5), *rec.get('dsi_significance_percent_decrease_NS_increase', [np.nan] * 3), *rec.get('dsi_direction_percent_decrease_increase', [np.nan] * 2)])
with (PROJECT_ROOT / '03_analysis/05_cover_combinations/data/class_and_example_timeseries.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['scope', 'combination', 'year', 'PV_pp', 'NPV_pp', 'BS_pp', 'main_DSI'])
    for g in range(27):
        for t, year in enumerate(years):
            writer.writerow(['class', g + 1, int(year), *fvc_series[g, t], dsi_series[g, t]])
    for e, g in enumerate(example_ids):
        for t, year in enumerate(years):
            writer.writerow(['example', g, int(year), *region_cover[e, t], region_dsi[e, t]])
print('Paired pixels', len(rows), 'top IDs', top_ids, 'example IDs', example_ids, flush=True)
print('Maximum |annual-series slope - reference trend|:', trend_check_max_abs_difference, flush=True)
for g in top_ids:
    print(class_records[g - 1], flush=True)
print('Examples', json.dumps(examples, ensure_ascii=False), flush=True)
