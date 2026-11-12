pass
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import rasterio
from tqdm import tqdm
ROOT = PROJECT_ROOT / '03_analysis/06_land_transitions'
OUT = PROJECT_ROOT / '03_analysis/06_land_transitions/data'
OUT.mkdir(parents=True, exist_ok=True)
FVC = PROJECT_ROOT / '03_analysis/05_cover_combinations/data'
LUCC = PROJECT_ROOT / '02_preprocessing/data/auxiliary/land_cover_0p1/class'
TOP_N, SMALL_CELL_N, ALPHA = (6, 30, 0.05)
names = ['Forest', 'Grassland', 'Cropland', 'Wetlands', 'Settlements', 'Other land']
names_cn = ['Forest', 'Grassland', 'Cropland', 'Wetlands', 'Settlements', 'Other land']
upstream = json.loads((PROJECT_ROOT / '03_analysis/05_cover_combinations/data/main_dsi_response_summary.json').read_text(encoding='utf-8'))
fvc_summary = json.loads((PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combination_summary.json').read_text(encoding='utf-8'))
assert upstream['dsi_alpha'] == ALPHA and upstream['fvc_primary_alpha'] == ALPHA
DSI = remap_path(upstream['main_dsi_root'])
z = np.load(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/paired_pixel_statistics.npz')
rows, cols, area = (z['rows'], z['cols'], z['area_km2'])
years = np.asarray(upstream['years'])
assert np.array_equal(years, np.arange(2001, 2022))
with rasterio.open(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/paired_fvc_combination.tif') as src:
    profile, shape, transform, crs = (src.profile.copy(), src.shape, src.transform, src.crs)
annual = np.zeros((len(rows), len(years)), np.uint8)
sources = []
source_paths = [FVC / x for x in ['paired_pixel_statistics.npz', 'fvc_pixel_statistics.npz', 'main_dsi_response_summary.json', 'fvc_combination_summary.json', 'paired_fvc_combination.tif']]
source_paths.extend([remap_path(upstream['dsi_trend_source']), remap_path(upstream['dsi_pvalue_source'])])
for j, year in enumerate(tqdm(years, desc='Read annual LUCC', unit='year')):
    path = LUCC / f'MCD12C1_LULC_CLASS_{year}.tif'
    with rasterio.open(path) as src:
        if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
            raise ValueError(f'LUCC grid mismatch: {path}')
        raster = src.read(1, masked=True)
        values = raster.data[rows, cols]
        good = ~np.ma.getmaskarray(raster)[rows, cols] & (values >= 1) & (values <= 6)
        annual[good, j] = values[good]
    source_paths.append(path)
endpoints = np.zeros((len(rows), 2), np.uint8)
for k, window in enumerate([annual[:, :3], annual[:, -3:]]):
    for c in range(1, 7):
        endpoints[np.sum(window == c, axis=1) >= 2, k] = c
valid = np.all(endpoints > 0, axis=1)
excluded = {}
for label, mask in [('early_uncertain', endpoints[:, 0] == 0), ('late_uncertain', endpoints[:, 1] == 0), ('either_uncertain', ~valid), ('both_uncertain', np.all(endpoints == 0, axis=1))]:
    excluded[label] = {'pixels': int(mask.sum()), 'area_km2': float(area[mask].sum()), 'area_percent_of_fvc_dsi_domain': float(area[mask].sum() / area.sum() * 100)}
indices = np.flatnonzero(valid)
start, end = endpoints[valid].T
transition = ((start.astype(int) - 1) * 6 + end - 1).astype(np.uint8)
annual = annual[valid]
area, rows, cols = (area[valid], rows[valid], cols[valid])
combination = z['combination'][valid]
fvc_trend, fvc_pvalue = (z['fvc_trend_pp_per_year'][valid], z['fvc_trend_pvalue'][valid])
trend = z['main_dsi_trend_per_year'][valid]
dsi_pvalue = z['main_dsi_trend_pvalue'][valid]
dsi_state = z['main_dsi_significance_state'][valid]
assert np.isfinite(trend).all() and np.isfinite(dsi_pvalue).all()
assert ((dsi_pvalue >= 0) & (dsi_pvalue <= 1)).all()
states_check = np.ones(fvc_trend.shape, dtype=np.uint8)
states_check[(fvc_pvalue < ALPHA) & (fvc_trend < 0)] = 0
states_check[(fvc_pvalue < ALPHA) & (fvc_trend > 0)] = 2
assert np.array_equal(combination, 9 * states_check[:, 0] + 3 * states_check[:, 1] + states_check[:, 2] + 1)
dsi_check = np.ones(len(area), dtype=np.uint8)
dsi_check[(dsi_pvalue < ALPHA) & (trend < 0)] = 0
dsi_check[(dsi_pvalue < ALPHA) & (trend > 0)] = 2
assert np.array_equal(dsi_state, dsi_check)
direction = np.where(trend < 0, 0, np.where(trend > 0, 2, 1)).astype(np.uint8)
top_combos = [int(x) for x in upstream['top_ids']]
assert top_combos == [14, 15, 17, 13, 23, 11, 19, 9]
fvc_display = np.full(len(area), len(top_combos), np.uint8)
for k, cid in enumerate(top_combos):
    fvc_display[combination == cid] = k
dsi = np.empty((len(area), len(years)), np.float64)
for j, year in enumerate(tqdm(years, desc='Read annual DSI', unit='year')):
    path = DSI / f'DI_{year}.tif'
    with rasterio.open(path) as src:
        if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
            raise ValueError(f'DSI grid mismatch: {path}')
        raster = src.read(1, masked=True)
        if np.ma.getmaskarray(raster)[rows, cols].any() or not np.isfinite(raster.data[rows, cols]).all():
            raise ValueError('Paired cache and source DSI mask disagree')
        dsi[:, j] = raster.data[rows, cols]
    source_paths.append(path)
time = years - years.mean()
fresh_trend = dsi @ time / np.sum(time ** 2)
max_slope_difference = float(np.max(np.abs(fresh_trend - trend)))
assert max_slope_difference < 1e-09, max_slope_difference
early, late = (dsi[:, :3].mean(axis=1), dsi[:, -3:].mean(axis=1))
delta = late - early
del dsi
counts = np.bincount(transition, minlength=36)
areas = np.bincount(transition, weights=area, minlength=36)
offdiag = np.flatnonzero((np.arange(36) // 6 != np.arange(36) % 6) & (counts > 0))
top_transitions = offdiag[np.argsort(-areas[offdiag], kind='stable')[:TOP_N]].tolist()
display_transitions = []
for origin in sorted({t // 6 for t in top_transitions}):
    if counts[origin * 7] > 0:
        display_transitions.append(origin * 7)
    display_transitions.extend((t for t in top_transitions if t // 6 == origin))
levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
metrics = np.column_stack([trend, early, late, delta])
records, cross_records = ([], [])
summary_q = np.full((36, 4, 5), np.nan)
cross_area = np.zeros((36, 27))
cross_count = np.zeros((36, 27), int)
cross_quantiles = np.full((36, 27, 5), np.nan)
display_area = np.zeros((36, len(top_combos) + 1))
display_count = np.zeros_like(display_area, dtype=int)
display_quantiles = np.full((*display_area.shape, 5), np.nan)
significance_percent = np.full((36, 3), np.nan)
direction_percent = np.full((36, 3), np.nan)
cross_significance = np.full((36, 27, 3), np.nan)
cross_direction = np.full((36, 27, 3), np.nan)
display_significance = np.full((*display_area.shape, 3), np.nan)
display_direction = np.full((*display_area.shape, 3), np.nan)
sig_tags = ['DSI_significant_decrease_percent', 'DSI_NS_percent', 'DSI_significant_increase_percent']
dir_tags = ['DSI_decrease_direction_percent', 'DSI_zero_direction_percent', 'DSI_increase_direction_percent']
for t in tqdm(range(36), desc='Summarize LUCC x FVC', unit='type'):
    chosen = transition == t
    weights = area[chosen]
    rec = {'transition_id': t + 1, 'origin_code': t // 6 + 1, 'destination_code': t % 6 + 1, 'origin': names[t // 6], 'destination': names[t % 6], 'pixels': int(counts[t]), 'area_km2': float(areas[t]), 'area_percent': float(areas[t] / area.sum() * 100)}
    if chosen.any():
        for m, metric in enumerate(['trend_per_year', 'early_DSI', 'late_DSI', 'delta_DSI']):
            values = metrics[chosen, m]
            order = np.argsort(values, kind='stable')
            midpoint = (np.cumsum(weights[order]) - weights[order] / 2) / weights.sum()
            q = np.interp(levels, midpoint, values[order])
            summary_q[t, m] = q
            for tag, value in zip(['q05', 'q25', 'q50', 'q75', 'q95'], q):
                rec[f'{metric}_{tag}'] = float(value)
            rec[f'{metric}_mean'] = float(np.average(values, weights=weights))
        for state in range(3):
            significance_percent[t, state] = weights[dsi_state[chosen] == state].sum() / weights.sum() * 100
            direction_percent[t, state] = weights[direction[chosen] == state].sum() / weights.sum() * 100
            rec[sig_tags[state]] = float(significance_percent[t, state])
            rec[dir_tags[state]] = float(direction_percent[t, state])
        rec['same_class_all21years_percent'] = float(weights[np.all(annual[chosen] == start[chosen, None], axis=1)].sum() / weights.sum() * 100)
    records.append(rec)
    for mode, group_values, n in [('all27', combination - 1, 27), ('display', fvc_display, len(top_combos) + 1)]:
        for g in range(n):
            cell = chosen & (group_values == g)
            ncell = int(cell.sum())
            cell_area = float(area[cell].sum())
            q = np.full(5, np.nan)
            sig, dirs = (np.full(3, np.nan), np.full(3, np.nan))
            if ncell:
                values, weights_cell = (trend[cell], area[cell])
                order = np.argsort(values, kind='stable')
                midpoint = (np.cumsum(weights_cell[order]) - weights_cell[order] / 2) / cell_area
                q = np.interp(levels, midpoint, values[order])
                for state in range(3):
                    sig[state] = weights_cell[dsi_state[cell] == state].sum() / cell_area * 100
                    dirs[state] = weights_cell[direction[cell] == state].sum() / cell_area * 100
            if mode == 'all27':
                cross_count[t, g], cross_area[t, g], cross_quantiles[t, g] = (ncell, cell_area, q)
                cross_significance[t, g], cross_direction[t, g] = (sig, dirs)
                label = f'C{g + 1:02d}'
            else:
                display_count[t, g], display_area[t, g], display_quantiles[t, g] = (ncell, cell_area, q)
                display_significance[t, g], display_direction[t, g] = (sig, dirs)
                label = f'C{top_combos[g]:02d}' if g < len(top_combos) else 'Other 19 pooled'
            row = {'grouping': mode, 'transition_id': t + 1, 'origin': names[t // 6], 'destination': names[t % 6], 'combination': label, 'pixels': ncell, 'small_cell_n_lt_30': bool(0 < ncell < SMALL_CELL_N), 'area_km2': cell_area, 'within_transition_percent': cell_area / areas[t] * 100 if areas[t] else None, 'domain_area_percent': cell_area / area.sum() * 100}
            for tag, value in zip(['q05', 'q25', 'q50', 'q75', 'q95'], q):
                row[f'trend_{tag}_per_year'] = float(value) if ncell else None
            for state in range(3):
                row[sig_tags[state]] = float(sig[state]) if ncell else None
                row[dir_tags[state]] = float(dirs[state]) if ncell else None
            cross_records.append(row)
order = np.argsort(np.abs(trend), kind='stable')
midpoint = (np.cumsum(area[order]) - area[order] / 2) / area.sum()
abs_q95 = float(np.interp(0.95, midpoint, np.abs(trend[order])) * 1000)
color_limit = float(max(0.2, np.ceil(abs_q95 / 0.2) * 0.2))
saturation = {'rule': 'area-weighted pixel abs annual DSI trend 95th percentile; ceil to 0.2 displayed units', 'display_unit': '10^-3 yr^-1', 'abs_q95_display_units': abs_q95, 'symmetric_limit': color_limit, 'pixel_area_beyond_limit_percent': float(area[np.abs(trend) * 1000 > color_limit].sum() / area.sum() * 100), 'all27_cell_median_area_beyond_limit_percent': float(cross_area[np.abs(cross_quantiles[:, :, 2]) * 1000 > color_limit].sum() / area.sum() * 100)}
fvc_cache = np.load(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_pixel_statistics.npz')
fvc_indices = z['fvc_input_indices'][valid]
assert np.array_equal(fvc_cache['rows'][fvc_indices], rows)
assert np.array_equal(fvc_cache['cols'][fvc_indices], cols)
sensitivity_records, sensitivity_summary = ([], [])
for alpha, labels_alpha in zip(fvc_cache['sensitivity_alphas'], fvc_cache['sensitivity_combinations']):
    ids = labels_alpha[fvc_indices]
    class_counts = np.bincount(ids, minlength=28)[1:]
    class_areas = np.bincount(ids, weights=area, minlength=28)[1:]
    agreement = float(area[ids == combination].sum() / area.sum() * 100)
    if np.isclose(alpha, ALPHA):
        assert np.array_equal(ids, combination)
    sensitivity_summary.append({'alpha': float(alpha), 'agreement_with_primary_percent': agreement, 'class_area_percent': (class_areas / area.sum() * 100).tolist()})
    for cid in range(1, 28):
        sensitivity_records.append({'alpha': float(alpha), 'combination': f'C{cid:02d}', 'pixels': int(class_counts[cid - 1]), 'area_km2': float(class_areas[cid - 1]), 'domain_area_percent': float(class_areas[cid - 1] / area.sum() * 100), 'agreement_with_primary_percent': agreement})
baseline_checks = {'historical_comparison': 'not run; only current v4 inputs are required'}
for path in tqdm(source_paths, desc='Fingerprint inputs', unit='file'):
    stat = path.stat()
    sources.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
np.savez_compressed(PROJECT_ROOT / '03_analysis/06_land_transitions/data/lucc_dsi_statistics.npz', counts=counts, area_km2=areas, summary_quantiles=summary_q, cross_area_km2=cross_area, cross_count=cross_count, cross_quantiles=cross_quantiles, display_area_km2=display_area, display_count=display_count, display_quantiles=display_quantiles, top_transitions=np.array(top_transitions), display_transitions=np.array(display_transitions), top_combinations=np.array(top_combos), significance_percent=significance_percent, direction_percent=direction_percent, cross_significance_percent=cross_significance, cross_direction_percent=cross_direction, display_significance_percent=display_significance, display_direction_percent=display_direction)
np.savez_compressed(PROJECT_ROOT / '03_analysis/06_land_transitions/data/paired_lucc_pixels.npz', rows=rows, cols=cols, area_km2=area, upstream_paired_indices=indices, origin=start, destination=end, transition=transition, combination=combination, FVC_trend_pp_per_year=fvc_trend, FVC_trend_pvalue=fvc_pvalue, DSI_trend_per_year=trend, DSI_trend_pvalue=dsi_pvalue, DSI_significance_state=dsi_state, DSI_direction_state=direction, early_DSI=early, late_DSI=late, delta_DSI=delta, LUCC_annual_classes=annual, years=years)
for filename, values in [('transition_classes.tif', transition + 1), ('origin_majority.tif', start), ('destination_majority.tif', end)]:
    raster = np.full(shape, 255, np.uint8)
    raster[rows, cols] = values
    profile.update(dtype='uint8', nodata=255, compress='deflate', count=1)
    with rasterio.open(OUT / filename, 'w', **profile) as dst:
        dst.write(raster, 1)
        dst.update_tags(PERIOD='2001-2021', ENDPOINT_RULE='At least 2 equal classes in each 3-year window', TRANSITION_ID='(origin-1)*6+destination; origin/destination codes 1..6')
for filename, table in [('transition_statistics.csv', records), ('transition_FVC_statistics.csv', cross_records), ('FVC_significance_sensitivity.csv', sensitivity_records)]:
    fields = list(dict.fromkeys((key for row in table for key in row)))
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
with (PROJECT_ROOT / '03_analysis/06_land_transitions/data/FVC_combination_legend.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['combination', 'PV_state', 'NPV_state', 'BS_state', 'alpha', 'state_definition'])
    arrows = ['↓', 'NS', '↑']
    for cid in range(1, 28):
        states = [(cid - 1) // 9, (cid - 1) % 9 // 3, (cid - 1) % 3]
        writer.writerow([f'C{cid:02d}', *[arrows[state] for state in states], ALPHA, 'Pixelwise OLS: decrease if slope<0 and p<alpha; NS if p>=alpha; increase if slope>0 and p<alpha'])
summary = {'schema_version': 4, 'status': 'significance_recomputed', 'years': years.tolist(), 'early_years': years[:3].tolist(), 'late_years': years[-3:].tolist(), 'classes': names, 'classes_cn': names_cn, 'paired_pixels': len(area), 'paired_area_km2': float(area.sum()), 'previous_paired_pixels': len(valid), 'excluded': excluded, 'unchanged_endpoint_area_percent': float(area[start == end].sum() / area.sum() * 100), 'top_transitions_zero_based': top_transitions, 'display_transitions_zero_based': display_transitions, 'top_changed_area_coverage_percent': float(areas[top_transitions].sum() / areas[offdiag].sum() * 100), 'top_combinations': top_combos, 'small_cell_flag_n': SMALL_CELL_N, 'selection': 'six largest off-diagonal LUCC transitions by area plus same-origin diagonal references; FVC focus from independent upstream FVC area', 'quantile_rule': 'area-weighted cumulative-weight midpoints; q05 q25 q50 q75 q95; not confidence intervals', 'FVC_source': str(FVC), 'FVC_alpha': ALPHA, 'DSI_alpha': ALPHA, 'FVC_trend_unit': 'percentage points per year', 'DSI_trend_unit': 'DSI per year', 'DSI_trend_display_unit': '10^-3 yr^-1', 'DSI_delta_display_unit': '10^-3; endpoint difference, not a rate', 'FVC_state_definition': 'significant decrease / NS / significant increase; raw two-sided OLS p<0.05; no FDR', 'DSI_trend_source': upstream['dsi_trend_source'], 'DSI_pvalue_source': upstream['dsi_pvalue_source'], 'DSI_annual_slope_vs_reference_max_abs_difference': max_slope_difference, 'color_scale': saturation, 'sensitivity': sensitivity_summary, 'baseline_comparison': baseline_checks, 'records': records, 'sources': sources}
(PROJECT_ROOT / '03_analysis/06_land_transitions/data/analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print('Common pixels:', len(area), '; focal combinations:', top_combos, '; color limit:', color_limit, flush=True)
print('Baseline:', baseline_checks, flush=True)
