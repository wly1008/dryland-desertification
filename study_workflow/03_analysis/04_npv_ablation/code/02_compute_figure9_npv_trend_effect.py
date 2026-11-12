pass
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, verify_source_record
from pathlib import Path
import csv
import json
import numpy as np
from pyproj import Geod
import rasterio
from scipy import stats
from tqdm import tqdm
ROOT = PROJECT_ROOT / '03_analysis/04_npv_ablation'
VERIFY_ROOT = PROJECT_ROOT / '02_preprocessing/data/auxiliary/sdg_0p1'
DATA_DIR = PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure9'
DATA_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR, END_YEAR = (2004, 2019)
YEARS = np.arange(START_YEAR, END_YEAR + 1)
PERIOD = f'{START_YEAR}_{END_YEAR}'
ALPHA = 0.05
SDG_THRESHOLD_FRACTION = 0.1
BOOTSTRAP_REPLICATES = 1999
BOOTSTRAP_BATCH = 100
RANDOM_SEED = 20260913
DIAGNOSTIC_SAMPLE_SIZE = 25000
DIAGNOSTIC_LAGS_DEG = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30]
BLOCK_CANDIDATES_DEG = [2, 5, 10, 15, 20, 30, 45, 60]
CORRELATION_CUTOFF = float(np.exp(-1))
NODATA = -9999.0
INPUT_ROOT = PROJECT_ROOT / '03_analysis/01_dsi/data/02DI'
INPUT_DIRS = {'without_npv': PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV/02study_period_weights', 'with_npv': PROJECT_ROOT / '03_analysis/01_dsi/data/02DI/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'}
PRODUCTS = ['FAO-WOCAT', 'JRC', 'Trends.Earth']
CLASS_NAMES = ['Significant degradation', 'No significant trend', 'Significant recovery']
TRANSITIONS = ['Unchanged category', 'NS to degradation', 'Recovery to NS', 'Recovery to degradation', 'Degradation to NS', 'NS to recovery', 'Degradation to recovery']
figure8_summary = json.loads((PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure8' / f'figure8_statistics_{PERIOD}.json').read_text(encoding='utf-8'))
if (figure8_summary['start_year'], figure8_summary['end_year']) != (START_YEAR, END_YEAR):
    raise ValueError('Figure 8 common-mask years differ from Figure 9.')
with rasterio.open(PROJECT_ROOT / '03_analysis/04_npv_ablation/data/figure8' / f'DSI_mean_with_NPV_{PERIOD}.tif') as source:
    masked_mean = source.read(1, masked=True)
    common_mask = ~np.ma.getmaskarray(masked_mean) & np.isfinite(masked_mean.data)
    profile, transform, crs, shape = (source.profile.copy(), source.transform, source.crs, source.shape)
rows, cols = np.where(common_mask)
n_pixels = len(rows)
if n_pixels != figure8_summary['common_valid_pixels'] or crs.to_epsg() != 4326:
    raise ValueError('Figure 8 mask or CRS differs from its record.')
manifest_lookup = {}
for item in figure8_summary['inputs']:
    manifest_lookup[item['group'], item['year']] = item
time_centered = YEARS.astype(float) - YEARS.mean()
time_ss = float(np.sum(time_centered ** 2))
degrees_of_freedom = len(YEARS) - 2
slopes = np.empty((2, n_pixels), dtype=np.float64)
pvalues = np.empty_like(slopes)
ols_checks = []
for variant, (group, directory) in enumerate(INPUT_DIRS.items()):
    annual_values = np.empty((len(YEARS), n_pixels), dtype=np.float64)
    for year_index, year in enumerate(tqdm(YEARS, desc=f'OLS inputs: {group}', unit='year')):
        path = directory / f'DI_{year}.tif'
        recorded = manifest_lookup[group, int(year)]
        verify_source_record(recorded)
        with rasterio.open(path) as source:
            if source.shape != shape or source.crs != crs or (not source.transform.almost_equals(transform)):
                raise ValueError(f'Grid mismatch: {path}')
            annual = source.read(1, masked=True)
            if np.any(np.ma.getmaskarray(annual)[common_mask]) or not np.isfinite(annual.data[common_mask]).all():
                raise ValueError(f'Common mask has missing years: {path}')
            annual_values[year_index] = annual.data[common_mask]
    centered = annual_values - annual_values.mean(axis=0)
    slopes[variant] = np.sum(time_centered[:, None] * centered, axis=0) / time_ss
    residuals = centered - time_centered[:, None] * slopes[variant]
    slope_se = np.sqrt(np.sum(residuals ** 2, axis=0) / degrees_of_freedom / time_ss)
    tvalues = np.divide(np.abs(slopes[variant]), slope_se, out=np.zeros(n_pixels), where=slope_se > 0)
    pvalues[variant] = 2 * stats.t.sf(tvalues, degrees_of_freedom)
    pvalues[variant][(slope_se == 0) & (slopes[variant] != 0)] = 0
    for pixel in np.linspace(0, n_pixels - 1, 48, dtype=int):
        reference_fit = stats.linregress(YEARS, annual_values[:, pixel])
        np.testing.assert_allclose(slopes[variant, pixel], reference_fit.slope, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(pvalues[variant, pixel], reference_fit.pvalue, rtol=1e-08, atol=1e-11)
    ols_checks.append({'variant': group, 'independent_scipy_linregress_samples': 48, 'passed': True})
    del annual_values, centered, residuals
old_directory = PROJECT_ROOT / '03_analysis/02_trends/data/Albedo_SMI_TGSI_Wet_LST_PV_NPV/02study_period_weights'
old_comparison = {}
for name, values in [(f'DI_trend_{PERIOD}.tif', slopes[1]), (f'DI_significance_pvalue_{PERIOD}.tif', pvalues[1])]:
    with rasterio.open(old_directory / name) as source:
        if source.shape != shape or source.crs != crs or (not source.transform.almost_equals(transform)):
            raise ValueError('Existing comparison trend grid mismatch.')
        old = source.read(1, masked=True)
        if np.any(np.ma.getmaskarray(old)[common_mask]):
            raise ValueError('Existing comparison trend has missing values within the common domain.')
        max_error = float(np.max(np.abs(old.data[common_mask] - values)))
        old_comparison[name] = {'max_absolute_difference': max_error}
        np.testing.assert_allclose(old.data[common_mask], values, rtol=2e-05, atol=8e-08)
classes = np.ones((2, n_pixels), dtype=np.uint8)
classes[(slopes > 0) & (pvalues < ALPHA)] = 0
classes[(slopes < 0) & (pvalues < ALPHA)] = 2
delta_slope = slopes[1] - slopes[0]
transition_codes = np.zeros(n_pixels, dtype=np.uint8)
for code, (before, after) in enumerate([(1, 0), (2, 1), (2, 0), (0, 1), (1, 2), (0, 2)], start=1):
    transition_codes[(classes[0] == before) & (classes[1] == after)] = code
assert np.array_equal(transition_codes != 0, classes[0] != classes[1])
geod = Geod(ellps='WGS84')
area_by_row = np.empty(shape[0])
for row in range(shape[0]):
    west, east = (transform.c, transform.c + transform.a)
    north, south = (transform.f + row * transform.e, transform.f + (row + 1) * transform.e)
    signed_area, _ = geod.polygon_area_perimeter([west, east, east, west], [north, north, south, south])
    area_by_row[row] = abs(signed_area) / 1000000.0
areas = area_by_row[rows]
matrix_area_pct = np.bincount(classes[0] * 3 + classes[1], weights=areas, minlength=9).reshape(3, 3) / areas.sum() * 100
transition_area_pct = np.bincount(transition_codes, weights=areas, minlength=7) / areas.sum() * 100
assert np.isclose(matrix_area_pct.sum(), 100) and np.isclose(transition_area_pct.sum(), 100)
profile.update(dtype='float32', nodata=NODATA, count=1, compress='deflate')
raster_values = {'DSI_trend_without_NPV': slopes[0], 'DSI_trend_with_NPV': slopes[1], 'DSI_pvalue_without_NPV': pvalues[0], 'DSI_pvalue_with_NPV': pvalues[1], 'DSI_trend_difference_with_minus_without_NPV': delta_slope, 'DSI_class_without_NPV': classes[0], 'DSI_class_with_NPV': classes[1], 'DSI_trend_category_transition': transition_codes}
for name, values in raster_values.items():
    grid = np.full(shape, NODATA, dtype=np.float32)
    grid[common_mask] = values
    with rasterio.open(DATA_DIR / f'{name}_{PERIOD}.tif', 'w', **profile) as destination:
        destination.write(grid, 1)
        destination.update_tags(START_YEAR=str(START_YEAR), END_YEAR=str(END_YEAR), YEAR_COUNT=str(len(YEARS)), ANALYSIS='OLS with two-sided t-test; nominal p<0.05; complete paired years', DIFFERENCE='with NPV minus without NPV', CLASS_CODES=json.dumps(CLASS_NAMES), TRANSITION_CODES=json.dumps(TRANSITIONS), MASK='same complete-case study mask as Figure 9')
external_values, external_class_vectors, product_metadata = ([], [], [])
external_valid = np.ones(n_pixels, dtype=bool)
for product in PRODUCTS:
    path = VERIFY_ROOT / 'resample_0.1deg_sum' / f'TrendsEarth_SDG15.3.1_2000-2023_{product}' / f'SDG_Indicator_15.3.1_2004-2019_sum_0.1deg.tif'
    with rasterio.open(path) as source:
        if source.shape != shape or source.crs != crs or (not source.transform.almost_equals(transform)):
            raise ValueError(f'External-product grid mismatch: {path}')
        external = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(external) & np.isfinite(external.data)
    minimum, maximum = (float(external.data[valid].min()), float(external.data[valid].max()))
    if not minimum < 0 < maximum:
        raise ValueError(f'SDG product does not include both negative and positive values: {product}')
    lower, upper = (minimum * SDG_THRESHOLD_FRACTION, maximum * SDG_THRESHOLD_FRACTION)
    external_values.append(-external.data[common_mask].astype(float) / maximum)
    reference_classes = np.ones(n_pixels, dtype=np.uint8)
    reference_classes[external.data[common_mask] < lower] = 0
    reference_classes[external.data[common_mask] > upper] = 2
    external_class_vectors.append(reference_classes)
    external_valid &= valid[common_mask]
    product_metadata.append({'product': product, 'path': str(path), 'size_bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'full_valid_min': minimum, 'full_valid_max': maximum, 'class_lower_threshold': lower, 'class_upper_threshold': upper, 'available_pixels_on_figure8_mask': int(np.count_nonzero(valid[common_mask]))})
validation_n = int(external_valid.sum())
external_y = np.asarray(external_values)[:, external_valid]
reference_classes = np.asarray(external_class_vectors)[:, external_valid]
validation_slopes = slopes[:, external_valid]
validation_classes = classes[:, external_valid]
validation_rows, validation_cols = (rows[external_valid], cols[external_valid])
validation_mask = np.zeros(shape, dtype=bool)
validation_mask[validation_rows, validation_cols] = True
grid = np.full(shape, NODATA, dtype=np.float32)
grid[common_mask] = external_valid.astype(float)
with rasterio.open(DATA_DIR / f'external_common_mask_{PERIOD}.tif', 'w', **profile) as destination:
    destination.write(grid, 1)
    destination.update_tags(VALUE_1='common to both DSI variants and all three SDG products', VALUE_0='valid DSI but missing at least one SDG product')
pearson = np.empty((3, 2))
balanced_agreement = np.empty((3, 2))
confusion = np.zeros((3, 2, 3, 3), dtype=np.int64)
standard_x = (validation_slopes - validation_slopes.mean(axis=1, keepdims=True)) / validation_slopes.std(axis=1, keepdims=True)
standard_y = (external_y - external_y.mean(axis=1, keepdims=True)) / external_y.std(axis=1, keepdims=True)
influence_fields, influence_names = ([], [])
for product_index, product in enumerate(PRODUCTS):
    individual_influences = []
    for variant in range(2):
        pearson[product_index, variant] = stats.pearsonr(validation_slopes[variant], external_y[product_index]).statistic
        confusion[product_index, variant] = np.bincount(reference_classes[product_index] * 3 + validation_classes[variant], minlength=9).reshape(3, 3)
        recall = np.diag(confusion[product_index, variant]) / confusion[product_index, variant].sum(axis=1)
        balanced_agreement[product_index, variant] = recall.mean()
        individual_influences.append(standard_x[variant] * standard_y[product_index] - pearson[product_index, variant] / 2 * (standard_x[variant] ** 2 + standard_y[product_index] ** 2))
    influence_fields.append(individual_influences[1] - individual_influences[0])
    influence_names.append(f'{product}: delta Pearson')
    class_share = np.bincount(reference_classes[product_index], minlength=3) / validation_n
    correct_difference = (validation_classes[1] == reference_classes[product_index]).astype(float) - (validation_classes[0] == reference_classes[product_index]).astype(float)
    class_delta = np.diag(confusion[product_index, 1] - confusion[product_index, 0]) / confusion[product_index, 0].sum(axis=1)
    influence_fields.append((correct_difference - class_delta[reference_classes[product_index]]) / (3 * class_share[reference_classes[product_index]]))
    influence_names.append(f'{product}: delta balanced agreement')
influence_fields = np.asarray(influence_fields)
rng = np.random.default_rng(RANDOM_SEED)
anchors = rng.choice(validation_n, size=min(DIAGNOSTIC_SAMPLE_SIZE, validation_n), replace=False)
index_grid = np.full(shape, -1, dtype=np.int32)
index_grid[validation_rows, validation_cols] = np.arange(validation_n)
diagnostic_rows = []
direction_correlations = np.full((6, 2, len(DIAGNOSTIC_LAGS_DEG)), np.nan)
for direction_index, direction in enumerate(['east', 'south']):
    for lag_index, lag in enumerate(DIAGNOSTIC_LAGS_DEG):
        step = round(lag / abs(transform.a if direction == 'east' else transform.e))
        shifted_rows = validation_rows[anchors] + (step if direction == 'south' else 0)
        shifted_cols = validation_cols[anchors] + (step if direction == 'east' else 0)
        inside = (shifted_rows < shape[0]) & (shifted_cols < shape[1])
        first = anchors[inside]
        second = index_grid[shifted_rows[inside], shifted_cols[inside]]
        paired = second >= 0
        first, second = (first[paired], second[paired])
        if len(first) < 200:
            continue
        lon1 = transform.c + (validation_cols[first] + 0.5) * transform.a
        lat1 = transform.f + (validation_rows[first] + 0.5) * transform.e
        lon2 = transform.c + (validation_cols[second] + 0.5) * transform.a
        lat2 = transform.f + (validation_rows[second] + 0.5) * transform.e
        _, _, distances = geod.inv(lon1, lat1, lon2, lat2)
        for field_index, field_name in enumerate(influence_names):
            correlation = float(stats.pearsonr(influence_fields[field_index, first], influence_fields[field_index, second]).statistic)
            direction_correlations[field_index, direction_index, lag_index] = correlation
            diagnostic_rows.append({'field': field_name, 'direction': direction, 'lag_degrees': lag, 'median_distance_km': float(np.median(distances) / 1000), 'paired_samples': len(first), 'correlation': correlation})
scale_records = []
for field_index, field_name in enumerate(influence_names):
    for direction_index, direction in enumerate(['east', 'south']):
        correlations = direction_correlations[field_index, direction_index]
        crossings = np.where((correlations[:-1] < CORRELATION_CUTOFF) & (correlations[1:] < CORRELATION_CUTOFF))[0]
        censored = len(crossings) == 0
        scale = DIAGNOSTIC_LAGS_DEG[int(crossings[0])] if not censored else DIAGNOSTIC_LAGS_DEG[-1]
        scale_records.append({'field': field_name, 'direction': direction, 'efolding_degrees': scale, 'range_censored': censored})
max_scale = max((item['efolding_degrees'] for item in scale_records))
requested_size = 2 * max_scale
eligible = [size for size in BLOCK_CANDIDATES_DEG if size >= requested_size]
if not eligible:
    raise ValueError('Diagnostic scale exceeds candidate blocks; extend BLOCK_CANDIDATES_DEG.')
main_block_size = float(eligible[0])
block_sizes = [main_block_size / 2, main_block_size, main_block_size * 2]
print(f'Validation n={validation_n:,}; max diagnostic scale={max_scale:g} deg; main blocks={main_block_size:g} deg', flush=True)
with (DATA_DIR / f'spatial_scale_diagnostics_{PERIOD}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
    writer.writeheader()
    writer.writerows(diagnostic_rows)
longitudes = transform.c + (validation_cols + 0.5) * transform.a
latitudes = transform.f + (validation_rows + 0.5) * transform.e
bootstrap_summary = []
bootstrap_output_arrays = {}
for block_size in block_sizes:
    block_x = np.floor((longitudes + 180) / block_size).astype(int)
    block_y = np.floor((latitudes + 90) / block_size).astype(int)
    block_labels = block_y * (int(np.ceil(360 / block_size)) + 1) + block_x
    unique_blocks, block_index = np.unique(block_labels, return_inverse=True)
    n_blocks = len(unique_blocks)
    if n_blocks < 30:
        raise ValueError(f'{block_size:g} degrees contains only {n_blocks} valid blocks; bootstrap estimates would be unstable.')
    block_counts = np.bincount(block_index, minlength=n_blocks)
    sufficient = np.zeros((n_blocks, 3, 27))
    for product_index in range(3):
        sufficient[:, product_index, 0] = block_counts
        for column, values in enumerate([standard_x[0], standard_x[1], standard_y[product_index], standard_x[0] ** 2, standard_x[1] ** 2, standard_y[product_index] ** 2, standard_x[0] * standard_y[product_index], standard_x[1] * standard_y[product_index]], start=1):
            sufficient[:, product_index, column] = np.bincount(block_index, weights=values, minlength=n_blocks)
        for variant in range(2):
            cell = reference_classes[product_index] * 3 + validation_classes[variant]
            sufficient[:, product_index, 9 + variant * 9:18 + variant * 9] = np.bincount(block_index * 9 + cell, minlength=n_blocks * 9).reshape(n_blocks, 9)
    original_totals = sufficient.sum(axis=0)
    for product_index in range(3):
        for variant in range(2):
            aggregate_r = original_totals[product_index, 7 + variant] / original_totals[product_index, 0]
            np.testing.assert_allclose(aggregate_r, pearson[product_index, variant], atol=1e-12)
            np.testing.assert_array_equal(original_totals[product_index, 9 + variant * 9:18 + variant * 9].reshape(3, 3), confusion[product_index, variant])
    draws_r = np.empty((BOOTSTRAP_REPLICATES, 3, 2))
    draws_ba = np.empty_like(draws_r)
    rng_boot = np.random.default_rng(RANDOM_SEED + round(block_size * 100))
    for start in tqdm(range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH), desc=f'Paired bootstrap {block_size:g} deg', unit='batch'):
        stop = min(start + BOOTSTRAP_BATCH, BOOTSTRAP_REPLICATES)
        multiplicities = rng_boot.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=stop - start)
        totals = (multiplicities @ sufficient.reshape(n_blocks, -1)).reshape(stop - start, 3, 27)
        counts = totals[:, :, 0]
        for variant in range(2):
            numerator = totals[:, :, 7 + variant] - totals[:, :, 1 + variant] * totals[:, :, 3] / counts
            xss = totals[:, :, 4 + variant] - totals[:, :, 1 + variant] ** 2 / counts
            yss = totals[:, :, 6] - totals[:, :, 3] ** 2 / counts
            draws_r[start:stop, :, variant] = numerator / np.sqrt(xss * yss)
            confusion_draws = totals[:, :, 9 + variant * 9:18 + variant * 9].reshape(stop - start, 3, 3, 3)
            reference_counts = confusion_draws.sum(axis=3)
            if np.any(reference_counts == 0):
                raise ValueError('A resample lacks a reference class; revise the block design or statistic.')
            draws_ba[start:stop, :, variant] = (np.diagonal(confusion_draws, axis1=2, axis2=3) / reference_counts).mean(axis=2)
    delta_r_draws = draws_r[:, :, 1] - draws_r[:, :, 0]
    delta_ba_draws = (draws_ba[:, :, 1] - draws_ba[:, :, 0]) * 100
    r_ci = np.quantile(delta_r_draws, [0.025, 0.975], axis=0).T
    ba_ci = np.quantile(delta_ba_draws, [0.025, 0.975], axis=0).T
    bootstrap_summary.append({'block_size_degrees': block_size, 'occupied_blocks': n_blocks, 'min_pixels_per_block': int(block_counts.min()), 'median_pixels_per_block': float(np.median(block_counts)), 'max_pixels_per_block': int(block_counts.max()), 'delta_r_ci95': r_ci.tolist(), 'delta_balanced_agreement_pp_ci95': ba_ci.tolist()})
    key = str(block_size).replace('.', 'p')
    bootstrap_output_arrays[f'delta_r_{key}deg'] = delta_r_draws
    bootstrap_output_arrays[f'delta_ba_pp_{key}deg'] = delta_ba_draws
    if block_size == main_block_size:
        main_r_ci, main_ba_ci = (r_ci, ba_ci)
np.savez_compressed(DATA_DIR / f'figure9_bootstrap_draws_{PERIOD}.npz', **bootstrap_output_arrays)
delta_r = pearson[:, 1] - pearson[:, 0]
delta_ba_pp = (balanced_agreement[:, 1] - balanced_agreement[:, 0]) * 100
threshold_sensitivity = []
for fraction in [0.05, 0.1, 0.15, 0.2]:
    for product_index, product in enumerate(PRODUCTS):
        raw = -external_y[product_index] * product_metadata[product_index]['full_valid_max']
        sensitivity_classes = np.ones(validation_n, dtype=np.uint8)
        sensitivity_classes[raw < fraction * product_metadata[product_index]['full_valid_min']] = 0
        sensitivity_classes[raw > fraction * product_metadata[product_index]['full_valid_max']] = 2
        agreements = []
        for variant in range(2):
            sensitivity_confusion = np.bincount(sensitivity_classes * 3 + validation_classes[variant], minlength=9).reshape(3, 3)
            agreements.append(float((np.diag(sensitivity_confusion) / sensitivity_confusion.sum(axis=1)).mean()))
        if fraction == SDG_THRESHOLD_FRACTION:
            np.testing.assert_allclose(agreements, balanced_agreement[product_index], atol=1e-14)
        threshold_sensitivity.append({'threshold_fraction': fraction, 'product': product, 'balanced_agreement_without': agreements[0], 'balanced_agreement_with': agreements[1], 'delta_balanced_agreement_pp': (agreements[1] - agreements[0]) * 100})
with (DATA_DIR / f'sdg_threshold_sensitivity_{PERIOD}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(threshold_sensitivity[0]))
    writer.writeheader()
    writer.writerows(threshold_sensitivity)
summary = {'start_year': START_YEAR, 'end_year': END_YEAR, 'year_count': len(YEARS), 'alpha': ALPHA, 'common_trend_pixels': n_pixels, 'common_area_km2': float(areas.sum()), 'validation_common_pixels': validation_n, 'validation_area_km2': float(areas[external_valid].sum()), 'class_names': CLASS_NAMES, 'transition_names': TRANSITIONS, 'transition_matrix_area_percent': matrix_area_pct.tolist(), 'transition_area_percent': transition_area_pct.tolist(), 'changed_area_percent': float(100 - transition_area_pct[0]), 'direction_reversal_area_percent': float(transition_area_pct[3] + transition_area_pct[6]), 'mean_delta_slope_area_weighted': float(np.average(delta_slope, weights=areas)), 'delta_slope_quantiles': np.quantile(delta_slope, [0, 0.005, 0.025, 0.5, 0.975, 0.995, 1]).tolist(), 'products': PRODUCTS, 'product_metadata': product_metadata, 'validation_weighting': 'equal pixel weights, same common mask for all products and both variants', 'pearson_r_direction_aligned': pearson.tolist(), 'delta_r': delta_r.tolist(), 'delta_r_ci95': main_r_ci.tolist(), 'balanced_agreement': balanced_agreement.tolist(), 'delta_balanced_agreement_pp': delta_ba_pp.tolist(), 'delta_balanced_agreement_pp_ci95': main_ba_ci.tolist(), 'confusion_matrices_counts': confusion.tolist(), 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'random_seed': RANDOM_SEED, 'main_block_size_degrees': main_block_size, 'bootstrap_sensitivity': bootstrap_summary, 'block_size_rule': 'candidate ceiling of twice the largest directional 1/e range of paired-metric influence fields; two consecutive lags below cutoff', 'spatial_scale_records': scale_records, 'ols_scipy_checks': ols_checks, 'old_trend_comparison': old_comparison, 'sdg_threshold_sensitivity': threshold_sensitivity, 'limitations': ['OLS significance is nominal, uncorrected for temporal autocorrelation or spatial multiple testing, matching section 3.1/3.2.', 'No significant DSI trend is not equivalent to ecological stability or the SDG middle threshold category.', 'SDG classes use the section 3.1 thresholds, not original reference truth labels.', 'Spatial-block intervals are approximate and conditional on the chosen external products and block design.', 'Three products are alternative references, not assumed mutually independent ground truth.']}
(DATA_DIR / f'figure9_statistics_{PERIOD}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
with (DATA_DIR / f'figure9_external_agreement_{PERIOD}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['product', 'n', 'r_without', 'r_with', 'delta_r', 'delta_r_ci_low', 'delta_r_ci_high', 'BA_without', 'BA_with', 'delta_BA_pp', 'delta_BA_pp_ci_low', 'delta_BA_pp_ci_high'])
    for k, product in enumerate(PRODUCTS):
        writer.writerow([product, validation_n, *pearson[k], delta_r[k], *main_r_ci[k], *balanced_agreement[k], delta_ba_pp[k], *main_ba_ci[k]])
with (DATA_DIR / f'figure9_category_transition_{PERIOD}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['without_NPV_class', 'with_NPV_class', 'percent_of_total_study_area'])
    for before in range(3):
        for after in range(3):
            writer.writerow([CLASS_NAMES[before], CLASS_NAMES[after], matrix_area_pct[before, after]])
print(json.dumps({key: summary[key] for key in ['changed_area_percent', 'direction_reversal_area_percent', 'delta_slope_quantiles', 'pearson_r_direction_aligned', 'delta_r', 'delta_r_ci95', 'delta_balanced_agreement_pp', 'delta_balanced_agreement_pp_ci95', 'main_block_size_degrees']}, indent=2), flush=True)
print(f'Figure 9 calculations saved: {DATA_DIR}')
