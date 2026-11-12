pass
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, study_boundary, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
from pathlib import Path
from itertools import product
import json
import csv
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pyproj import Geod
from scipy.stats import t as student_t
from tqdm import tqdm
ROOT = PROJECT_ROOT / '03_analysis/05_cover_combinations'
OUT = PROJECT_ROOT / '03_analysis/05_cover_combinations/data'
OUT.mkdir(parents=True, exist_ok=True)
COMPONENT_ROOT = PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc'
STUDY = study_boundary()
YEARS = np.arange(2001, 2022)
COMPONENTS = ['PV', 'NPV', 'BS']
PRIMARY_ALPHA = 0.05
SENSITIVITY_ALPHAS = [0.01, 0.05, 0.1]
STATE_NAMES = ['significant decrease', 'not significant', 'significant increase']
with rasterio.open(PROJECT_ROOT / '02_preprocessing/data/non_growing_season_2001_2021/fvc/PV/PV_2001.tif') as src:
    profile, shape, transform, crs = (src.profile.copy(), src.shape, src.transform, src.crs)
study = gpd.read_file(STUDY).to_crs(crs)
study_mask = geometry_mask(list(study.geometry), out_shape=shape, transform=transform, invert=True)
study_rows, study_cols = np.where(study_mask)
values = np.empty((len(study_rows), len(YEARS), 3), np.float32)
valid = np.ones(len(study_rows), bool)
sources = []
tasks = [(t, y, c, name) for t, y in enumerate(YEARS) for c, name in enumerate(COMPONENTS)]
for t, year, c, name in tqdm(tasks, desc='Read FVC annual means', unit='raster'):
    path = COMPONENT_ROOT / name / f'{name}_{year}.tif'
    with rasterio.open(path) as src:
        if src.shape != shape or src.crs != crs or (not src.transform.almost_equals(transform)):
            raise ValueError(f'Grid mismatch: {path}')
        a = src.read(1, masked=True)
        valid &= (~np.ma.getmaskarray(a) & np.isfinite(a.data))[study_mask]
        v = a.data[study_mask].astype(np.float32) / 100.0
        values[:, t, c] = 100 - v if c < 2 else v
    sources.append({'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
series = np.ascontiguousarray(values[valid])
del values
rows, cols = (study_rows[valid], study_cols[valid])
geod = Geod(ellps='WGS84')
row_area = np.empty(shape[0])
for row in range(shape[0]):
    top, bottom = (transform.f + row * transform.e, transform.f + (row + 1) * transform.e)
    area, _ = geod.polygon_area_perimeter([transform.c, transform.c + transform.a, transform.c + transform.a, transform.c], [top, top, bottom, bottom])
    row_area[row] = abs(area) / 1000000.0
weights = row_area[rows]
time = YEARS - YEARS.mean()
sxx = np.sum(time ** 2)
means = series.mean(axis=1, dtype=np.float64)
trend = np.einsum('t,ntc->nc', time, series, optimize=True) / sxx
fitted = means[:, None, :] + time[None, :, None] * trend[:, None, :]
residual = series - fitted
sse = np.sum(residual ** 2, axis=1, dtype=np.float64)
residual_sd = np.sqrt(sse / (len(YEARS) - 2))
slope_se = residual_sd / np.sqrt(sxx)
t_stat = np.divide(trend, slope_se, out=np.zeros_like(trend), where=slope_se > 0)
p_value = 2 * student_t.sf(np.abs(t_stat), df=len(YEARS) - 2)
p_value[(slope_se == 0) & (trend != 0)] = 0.0
p_value[(slope_se == 0) & (trend == 0)] = 1.0
del fitted, residual
combos = list(product(range(3), repeat=3))
records = []
case_labels = []
for alpha in SENSITIVITY_ALPHAS:
    states = np.ones(trend.shape, np.uint8)
    significant = p_value < alpha
    states[significant & (trend < 0)] = 0
    states[significant & (trend > 0)] = 2
    labels = (states[:, 0] * 9 + states[:, 1] * 3 + states[:, 2] + 1).astype(np.uint8)
    case_labels.append(labels)
    if alpha == PRIMARY_ALPHA:
        primary_labels = labels.copy()
    grid = np.full(shape, 255, np.uint8)
    grid[study_mask] = 0
    grid[rows, cols] = labels
    profile.update(dtype='uint8', nodata=255, count=1, compress='deflate')
    name = 'fvc_combination_2001_2021.tif' if alpha == PRIMARY_ALPHA else f'fvc_combination_alpha_{alpha:g}.tif'
    with rasterio.open(OUT / name, 'w', **profile) as dst:
        dst.write(grid, 1)
        dst.update_tags(METHOD='Pixelwise OLS slope and two-sided Student t test; no ternary closure', ALPHA=str(alpha), DF=str(len(YEARS) - 2), MULTIPLE_TESTING='none', CODES='1..27=PV*9+NPV*3+BS+1; states sig decrease=0, NS=1, sig increase=2; 0=incomplete; 255=outside')
    class_counts = np.bincount(labels, minlength=28)[1:]
    class_area = np.bincount(labels, weights=weights, minlength=28)[1:] / weights.sum() * 100
    records.append({'alpha': alpha, 'class_area_percent': class_area.tolist(), 'class_counts': class_counts.tolist(), 'component_state_area_percent': [[float(weights[states[:, c] == state].sum() / weights.sum() * 100) for state in range(3)] for c in range(3)]})
for c, name in enumerate(COMPONENTS):
    grid = np.full(shape, -9999, np.float32)
    grid[rows, cols] = trend[:, c]
    profile.update(dtype='float32', nodata=-9999, count=1, compress='deflate')
    with rasterio.open(OUT / f'{name}_trend_pp_per_year.tif', 'w', **profile) as dst:
        dst.write(grid, 1)
        dst.update_tags(UNIT='percentage points per year', METHOD='OLS 2001-2021')
    grid = np.full(shape, -9999, np.float32)
    grid[rows, cols] = p_value[:, c]
    with rasterio.open(OUT / f'{name}_trend_pvalue.tif', 'w', **profile) as dst:
        dst.write(grid, 1)
        dst.update_tags(UNIT='probability', METHOD='two-sided Student t test for OLS slope', DF=str(len(YEARS) - 2))
area = np.bincount(primary_labels, weights=weights, minlength=28)[1:] / weights.sum() * 100
counts = np.bincount(primary_labels, minlength=28)[1:]
for case, case_ids in zip(records, case_labels):
    case['agreement_with_primary_area_percent'] = float(weights[case_ids == primary_labels].sum() / weights.sum() * 100)
np.save(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_annual_cover_pp.npy', series)
np.savez_compressed(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_pixel_statistics.npz', rows=rows.astype(np.int16), cols=cols.astype(np.int16), area_km2=weights, trend_pp_per_year=trend, trend_pvalue=p_value, mean_cover_pp=means, residual_sd_pp=residual_sd, slope_se_pp_per_year=slope_se, combination=primary_labels, sensitivity_alphas=np.asarray(SENSITIVITY_ALPHAS), sensitivity_combinations=np.asarray(case_labels), years=YEARS)
summary = {'years': YEARS.tolist(), 'component_root': str(COMPONENT_ROOT), 'study_path': STUDY.relative_to(PROJECT_ROOT).as_posix(), 'components': COMPONENTS, 'study_pixels': len(study_rows), 'complete_fvc_pixels': len(rows), 'area_km2': float(weights.sum()), 'primary_alpha': PRIMARY_ALPHA, 'method': 'Pixelwise OLS annual trends; two-sided Student t test for slope; raw p<alpha; no multiple-testing correction', 'degrees_of_freedom': len(YEARS) - 2, 'classification': 'significant decrease if slope<0 and p<alpha; NS if p>=alpha; significant increase if slope>0 and p<alpha', 'normalization': 'stored PV/NPV direction and scale restored; original component cover retained; no closure or S>0 screening', 'combinations': [{'id': i + 1, 'states': list(states), 'description': '; '.join([f'{c} {STATE_NAMES[state]}' for c, state in zip(COMPONENTS, states)]), 'pixels': int(counts[i]), 'area_percent': float(area[i])} for i, states in enumerate(combos)], 'sensitivity': records, 'sources': sources}
(PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combination_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
with (PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_combinations_all27.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['id', 'PV_state', 'NPV_state', 'BS_state', 'pixels', 'area_percent'])
    for rec in summary['combinations']:
        writer.writerow([rec['id'], *[STATE_NAMES[state] for state in rec['states']], rec['pixels'], rec['area_percent']])
with (PROJECT_ROOT / '03_analysis/05_cover_combinations/data/fvc_significance_sensitivity_all27.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['alpha', 'combination', 'pixels', 'area_percent', 'whole_domain_agreement_with_alpha_0.05_percent'])
    for case in records:
        for i in range(27):
            writer.writerow([case['alpha'], i + 1, case['class_counts'][i], case['class_area_percent'][i], case['agreement_with_primary_area_percent']])
print('Complete FVC pixels:', len(rows), flush=True)
for i in np.argsort(-area)[:10]:
    print(summary['combinations'][i], flush=True)
