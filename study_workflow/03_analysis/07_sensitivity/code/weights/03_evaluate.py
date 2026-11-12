"""Two time windows, fixed domains, OLS, NPV contrasts, paired external metrics."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import numpy as np
import rasterio
from scipy import stats
import config as c
from common import area_percentages, assert_grid, cell_areas, file_record, finish_stage, ols, paired_external, read_json, trend_classes, valid_array, write_csv, write_json, write_raster

def load_series(scenario, variant, years, mask, reference):
    values = np.empty((len(years), int(mask.sum())), dtype=np.float64)
    for i, year in enumerate(years):
        with rasterio.open(c.dsi_path(scenario, variant, year)) as source:
            assert_grid(source, reference)
            raw, valid = valid_array(source)
        if np.any(mask & ~valid):
            raise ValueError(f'Incomplete fixed domain: {scenario} {variant} {year}')
        values[i] = raw[mask]
    return values

def checked_fit(values, years, label):
    slope, p = ols(values, years)
    for index in np.linspace(0, values.shape[1] - 1, 48, dtype=int):
        fit = stats.linregress(years, values[:, index])
        np.testing.assert_allclose(slope[index], fit.slope, rtol=1e-10, atol=1e-14, err_msg=label)
        np.testing.assert_allclose(p[index], fit.pvalue, rtol=1e-08, atol=1e-11, err_msg=label)
    return (slope, p)

def main():
    out = c.OUTPUT_ROOT
    fig8 = read_json(c.ABLATION_ROOT / 'data' / 'figure8' / 'figure8_statistics_2004_2019.json')
    fig9 = read_json(c.ABLATION_ROOT / 'data' / 'figure9' / 'figure9_statistics_2004_2019.json')
    with rasterio.open(c.original_dsi_path('with_NPV', 2001)) as source:
        reference = {'shape': source.shape, 'crs': source.crs, 'transform': source.transform}
        profile = source.profile.copy()
    records = []

    def load_reference(path):
        with rasterio.open(path) as source:
            assert_grid(source, reference)
            raw, valid = valid_array(source)
        records.append(file_record(path))
        return (raw, valid)
    old_slope, old_valid = load_reference(c.LONG_TREND_ROOT / 'DI_trend_2001_2021.tif')
    old_p, old_pvalid = load_reference(c.LONG_TREND_ROOT / 'DI_significance_pvalue_2001_2021.tif')
    long_mask = old_valid & old_pvalid
    if int(long_mask.sum()) != c.EXPECTED_LONG_PIXELS:
        raise ValueError('Original long-term domain changed')
    long_rows, _ = np.where(long_mask)
    row_latitudes = reference['transform'].f + (np.arange(reference['shape'][0]) + 0.5) * reference['transform'].e
    long_areas = np.cos(np.deg2rad(row_latitudes[long_rows]))
    old_long_class = trend_classes(old_slope[long_mask], old_p[long_mask])
    _, ablation_mask = load_reference(c.ABLATION_ROOT / 'data' / 'figure8' / 'DSI_mean_with_NPV_2004_2019.tif')
    if int(ablation_mask.sum()) != c.EXPECTED_ABLATION_PIXELS:
        raise ValueError('Original ablation domain changed')
    rows, cols = np.where(ablation_mask)
    areas = cell_areas(reference['shape'], reference['transform'])[rows]
    external_y, reference_classes, metadata = ([], [], [])
    external_valid = np.ones(len(rows), dtype=bool)
    for product in c.PRODUCTS:
        raw, valid = load_reference(c.external_path(product))
        minimum, maximum = (float(raw[valid].min()), float(raw[valid].max()))
        lower, upper = (minimum * c.SDG_THRESHOLD_FRACTION, maximum * c.SDG_THRESHOLD_FRACTION)
        y = raw[ablation_mask]
        classes = np.ones(len(rows), dtype=np.uint8)
        classes[y < lower] = 0
        classes[y > upper] = 2
        external_y.append(-y.astype(float) / maximum)
        reference_classes.append(classes)
        external_valid &= valid[ablation_mask]
        metadata.append({'product': product, 'full_min': minimum, 'full_max': maximum, 'lower_threshold': lower, 'upper_threshold': upper})
    if int(external_valid.sum()) != c.EXPECTED_EXTERNAL_PIXELS:
        raise ValueError('Original external common domain changed')
    external_y = np.asarray(external_y)[:, external_valid]
    reference_classes = np.asarray(reference_classes)[:, external_valid]
    transform = reference['transform']
    lon = transform.c + (cols[external_valid] + 0.5) * transform.a
    lat = transform.f + (rows[external_valid] + 0.5) * transform.e
    bx = np.floor((lon + 180) / c.BLOCK_SIZE_DEG).astype(int)
    by = np.floor((lat + 90) / c.BLOCK_SIZE_DEG).astype(int)
    labels = by * (int(np.ceil(360 / c.BLOCK_SIZE_DEG)) + 1) + bx
    blocks, block_index = np.unique(labels, return_inverse=True)
    n_blocks = len(blocks)
    external_grid_mask = np.zeros(reference['shape'], dtype=bool)
    external_grid_mask[rows[external_valid], cols[external_valid]] = True
    for name, mask in [('long_term_2001_2021', long_mask), ('npv_contrast_2004_2019', ablation_mask), ('external_common_2004_2019', external_grid_mask)]:
        write_raster(out / 'domains' / f'{name}.tif', np.ones(int(mask.sum()), dtype=np.uint8), mask, profile, {'FIXED_DOMAIN': name}, categorical=True)
    long_records, contrast_records, external_records, all_results = ([], [], [], {})
    baseline_long_class = None
    audit = {'long_pixels': int(long_mask.sum()), 'ablation_pixels': len(rows), 'external_pixels': int(external_valid.sum()), 'bootstrap_blocks': n_blocks, 'independent_ols_checks_per_fit': 48, 'external_product_metadata': metadata}
    for scenario in c.SCENARIOS:
        print(f'Evaluate {scenario}: 2001-2021 trends', flush=True)
        scenario_root = out / 'scenarios' / scenario
        long_values = load_series(scenario, 'with_NPV', c.YEARS, long_mask, reference)
        long_slope, long_p = checked_fit(long_values, c.YEARS, scenario)
        long_class = trend_classes(long_slope, long_p)
        del long_values
        if scenario == c.BASELINE:
            np.testing.assert_allclose(long_slope, old_slope[long_mask], rtol=2e-05, atol=8e-08)
            np.testing.assert_allclose(long_p, old_p[long_mask], rtol=2e-05, atol=8e-08)
            np.testing.assert_array_equal(long_class, old_long_class)
            baseline_long_class = long_class.copy()
        area_pct = area_percentages(long_class, long_areas)
        agreement = float(np.sum(long_areas[long_class == baseline_long_class]) / long_areas.sum() * 100)
        long_result = {'scenario': scenario, 'period': '2001-2021', 'pixels': int(long_mask.sum()), 'area_weighting': 'cos(latitude); Figure 7', 'degradation_area_percent': float(area_pct[0]), 'no_significant_trend_area_percent': float(area_pct[1]), 'recovery_area_percent': float(area_pct[2]), 'class_agreement_with_G15_area_percent': agreement}
        long_records.append(long_result)
        for name, values in [('DSI_slope', long_slope), ('DSI_pvalue', long_p), ('DSI_class', long_class)]:
            write_raster(scenario_root / 'trends_2001_2021' / f'{name}.tif', values, long_mask, profile, {'SCENARIO': scenario, 'PERIOD': '2001-2021', 'ANALYSIS': 'OLS; nominal two-sided Student t-test; P<0.05', 'CLASS_NAMES': str(c.CLASS_NAMES)}, categorical=name.endswith('class'))
        print(f'Evaluate {scenario}: 2004-2019 NPV contrast', flush=True)
        slopes, pvalues, classes, means = ([], [], [], [])
        for variant in c.VARIANTS:
            values = load_series(scenario, variant, c.ABLATION_YEARS, ablation_mask, reference)
            slope, p = checked_fit(values, c.ABLATION_YEARS, f'{scenario} {variant}')
            means.append(values.mean(axis=0))
            slopes.append(slope)
            pvalues.append(p)
            classes.append(trend_classes(slope, p))
            del values
        slopes, pvalues, classes, means = map(np.asarray, (slopes, pvalues, classes, means))
        matrix = np.bincount(classes[0] * 3 + classes[1], weights=areas, minlength=9).reshape(3, 3) / areas.sum() * 100
        changed = classes[0] != classes[1]
        reversal = (classes[0] == 0) & (classes[1] == 2) | (classes[0] == 2) & (classes[1] == 0)
        delta = means[1] - means[0]
        contrast_result = {'scenario': scenario, 'period': '2004-2019', 'pixels': len(rows), 'area_km2': float(areas.sum()), 'area_weighting': 'WGS84 ellipsoidal pixel areas; Figures 8-9', 'mean_delta_DSI': float(np.average(delta, weights=areas)), 'state_increase_area_percent': float(areas[delta > 0].sum() / areas.sum() * 100), 'changed_trend_class_area_percent': float(areas[changed].sum() / areas.sum() * 100), 'direct_reversal_area_percent': float(areas[reversal].sum() / areas.sum() * 100), 'mean_delta_slope': float(np.average(slopes[1] - slopes[0], weights=areas))}
        contrast_records.append(contrast_result)
        matrix_rows = [{'without_NPV_class': c.CLASS_NAMES[i], 'with_NPV_class': c.CLASS_NAMES[j], 'paired_total_area_percent': float(matrix[i, j])} for i in range(3) for j in range(3)]
        write_csv(scenario_root / 'trend_class_transition_2004_2019.csv', matrix_rows)
        for i, variant in enumerate(c.VARIANTS):
            for name, vector in [('DSI_slope', slopes[i]), ('DSI_pvalue', pvalues[i]), ('DSI_class', classes[i]), ('DSI_mean', means[i])]:
                write_raster(scenario_root / 'contrast_2004_2019' / f'{name}_{variant}.tif', vector, ablation_mask, profile, {'SCENARIO': scenario, 'VARIANT': variant, 'PERIOD': '2004-2019'}, categorical=name.endswith('class'))
        write_raster(scenario_root / 'contrast_2004_2019' / 'delta_DSI_with_minus_without_NPV.tif', delta, ablation_mask, profile, {'DIFFERENCE': 'with NPV minus without NPV'})
        write_raster(scenario_root / 'contrast_2004_2019' / 'delta_slope_with_minus_without_NPV.tif', slopes[1] - slopes[0], ablation_mask, profile)
        if scenario == c.BASELINE:
            np.testing.assert_allclose(contrast_result['mean_delta_DSI'], fig8['delta_mean'], atol=1e-12)
            np.testing.assert_allclose(contrast_result['changed_trend_class_area_percent'], fig9['changed_area_percent'], atol=1e-09)
            np.testing.assert_allclose(matrix, fig9['transition_matrix_area_percent'], atol=1e-09)
        print(f'Evaluate {scenario}: paired spatial bootstrap, {n_blocks} blocks x {c.BOOTSTRAP_REPLICATES} replicates', flush=True)
        external = paired_external(slopes[:, external_valid], classes[:, external_valid], external_y, reference_classes, block_index, n_blocks, save_draws=scenario_root / 'paired_bootstrap_draws_2004_2019.npz')
        if scenario == c.BASELINE:
            np.testing.assert_allclose(external['r'], fig9['pearson_r_direction_aligned'], atol=1e-12)
            np.testing.assert_allclose(external['delta_r_ci95'], fig9['delta_r_ci95'], atol=1e-12)
            np.testing.assert_allclose(external['delta_ba_pp_ci95'], fig9['delta_balanced_agreement_pp_ci95'], atol=1e-10)
        for i, product in enumerate(c.PRODUCTS):
            external_records.append({'scenario': scenario, 'product': product, 'pixels': external['pixels'], 'r_without_NPV': external['r'][i][0], 'r_with_NPV': external['r'][i][1], 'delta_r': external['delta_r'][i], 'delta_r_ci95_low': external['delta_r_ci95'][i][0], 'delta_r_ci95_high': external['delta_r_ci95'][i][1], 'BA_without_NPV_percent': external['ba_percent'][i][0], 'BA_with_NPV_percent': external['ba_percent'][i][1], 'delta_BA_pp': external['delta_ba_pp'][i], 'delta_BA_ci95_low_pp': external['delta_ba_pp_ci95'][i][0], 'delta_BA_ci95_high_pp': external['delta_ba_pp_ci95'][i][1]})
        all_results[scenario] = {'long_term': long_result, 'NPV_contrast': contrast_result, 'transition_matrix_area_percent': matrix.tolist(), 'external': external}
        write_json(scenario_root / 'statistics.json', all_results[scenario])
    tables = out / 'tables'
    write_csv(tables / 'long_term_trends.csv', long_records)
    write_csv(tables / 'NPV_contrasts.csv', contrast_records)
    write_csv(tables / 'external_consistency.csv', external_records)
    write_json(out / 'all_statistics.json', all_results)
    write_json(out / 'audit' / 'evaluation_sources.json', records)
    write_json(out / 'audit' / 'evaluation_checks.json', audit)
    finish_stage('03_evaluate', **audit, baseline_existing_results_reproduced=True)
    print('Evaluation complete; baseline Fig.7-9 comparisons passed.', flush=True)
if __name__ == '__main__':
    main()
