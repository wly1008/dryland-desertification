"""Reproduce A and compute only D in the existing support-material workspace."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, source_record, verify_source_record, invalidate_completion_records
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import rasterio
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import trend_config as c
from trend_numeric import ols, sen_mk, bh_adjust, states, area_fractions, weighted_quantile, paired_external

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def signature():
    digest = hashlib.sha256()
    for filename in ['trend_config.py', 'trend_numeric.py', '03_compute_trend_D.py']:
        digest.update((c.CODE_ROOT / filename).read_bytes())
    return digest.hexdigest()

def main():
    for folder in ['audit', 'maps', 'vectors', 'tables']:
        (c.OUT / folder).mkdir(parents=True, exist_ok=True)
    complete = c.OUT / 'audit' / 'complete.json'
    if complete.exists():
        status = json.loads(complete.read_text(encoding='utf-8'))
        try:
            if os.environ.get('REPRO_RECOMPUTE_SENSITIVITY') == '1':
                raise ValueError('Explicit recomputation requested')
            if status['code_signature'] != signature():
                raise ValueError('Trend analysis code changed')
            for item in status['sources']:
                verify_source_record(item)
        except (ValueError, FileNotFoundError) as error:
            invalidate_completion_records([complete], str(error))
            print('Recompute trend sensitivity: ' + str(error), flush=True)
        else:
            print('Reuse completed A-versus-D analysis', flush=True)
            return
    sources = {}

    def record(path):
        path = remap_path(path)
        sources[str(path)] = source_record(path)

    def read_json(path):
        record(path)
        return json.loads(remap_path(path).read_text(encoding='utf-8'))
    reference_path = c.annual_path('with_NPV', 2001)
    with rasterio.open(reference_path) as source:
        shape, transform, crs, profile = (source.shape, source.transform, source.crs, source.profile.copy())

    def raster(path):
        record(path)
        with rasterio.open(path) as source:
            if source.shape != shape or source.crs != crs or (not source.transform.almost_equals(transform)):
                raise ValueError('Grid mismatch: ' + str(path))
            values = source.read(1, masked=True)
        return (values.data, ~np.ma.getmaskarray(values) & np.isfinite(values.data))

    def write_map(path, values, rows, cols, kind, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        categorical = kind == 'state'
        dtype, nodata = ('uint8', 255) if categorical else ('float32', -9999)
        grid = np.full(shape, nodata, dtype=dtype)
        grid[rows, cols] = values
        options = profile.copy()
        options.pop('nbits', None)
        options.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress='deflate', tiled=True, blockxsize=256, blockysize=256, predictor=2 if categorical else 3)
        with rasterio.open(path, 'w', **options) as target:
            target.write(grid, 1)
            target.update_tags(**{key: str(value) for key, value in metadata.items()})
        with rasterio.open(path) as source:
            np.testing.assert_array_equal(source.read(1)[rows, cols], grid[rows, cols])

    def series(variant, years, mask):
        values = np.empty((len(years), int(mask.sum())), dtype=np.float64)
        for index, year in enumerate(years):
            raw, valid = raster(c.annual_path(variant, year))
            if np.any(mask & ~valid):
                raise ValueError('Missing annual values inside fixed domain')
            values[index] = raw[mask]
        return values
    datasets = {}
    statistics = {}
    numeric_checks = []

    def process(key, variable, years, values, rows, cols, weights, area_label, a_slope=None, a_p=None):
        n = values.shape[1]
        if a_slope is None:
            a_slope, a_p = ols(values, years)
        indices = np.unique(np.linspace(0, n - 1, 48, dtype=int))
        for index in indices:
            sample = np.asarray(values[:, index], dtype=float)
            if np.ptp(sample) == 0:
                np.testing.assert_allclose(a_slope[index], 0, atol=1e-12)
                np.testing.assert_allclose(a_p[index], 1, atol=1e-12)
                continue
            fit = stats.linregress(years, sample)
            np.testing.assert_allclose(a_slope[index], fit.slope, rtol=2e-05, atol=8e-08)
            np.testing.assert_allclose(a_p[index], fit.pvalue, rtol=2e-05, atol=8e-08)
        slope = np.empty(n)
        p = np.empty(n)
        mk_s = np.empty(n, dtype=np.int32)
        variance = np.empty(n)
        for start in tqdm(range(0, n, c.CHUNK_PIXELS), desc=f'D: {key}', unit='chunk'):
            stop = min(start + c.CHUNK_PIXELS, n)
            slope[start:stop], p[start:stop], mk_s[start:stop], variance[start:stop] = sen_mk(values[:, start:stop], years)
        adjusted = bh_adjust(p)
        np.testing.assert_allclose(adjusted, multipletests(p, method='fdr_bh')[1], rtol=0, atol=2e-15)
        for index in indices:
            reference_sen = stats.theilslopes(np.asarray(values[:, index], dtype=float), years).slope
            np.testing.assert_allclose(slope[index], reference_sen, rtol=1e-12, atol=1e-13)
        a_state = states(a_slope, a_p, c.ALPHA)
        d_state = states(slope, adjusted, c.ALPHA)
        direction_agree = float(np.sum(weights[np.sign(a_slope) == np.sign(slope)]) / weights.sum() * 100)
        class_agree = float(np.sum(weights[a_state == d_state]) / weights.sum() * 100)
        stats_by_method = {}
        for method, current_slope, current_state in [('A_OLS', a_slope, a_state), ('D_Sen_MK_BH', slope, d_state)]:
            fractions = area_fractions(current_state, weights)
            stats_by_method[method] = {'decrease_area_percent': float(fractions[0]), 'no_significant_trend_area_percent': float(fractions[1]), 'increase_area_percent': float(fractions[2]), 'class_agreement_with_A_area_percent': 100.0 if method == 'A_OLS' else class_agree, 'direction_agreement_with_A_area_percent': 100.0 if method == 'A_OLS' else direction_agree}
        statistics[key] = {'variable': variable, 'period': f'{years[0]}-{years[-1]}', 'pixels': n, 'BH_family_size': n, 'area_weighting': area_label, 'methods': stats_by_method, 'zero_sen_slope_and_BH_significant_pixels': int(np.sum((slope == 0) & (adjusted < c.ALPHA))), 'tied_series_pixels': int(np.sum(variance < len(years) * (len(years) - 1) * (2 * len(years) + 5) / 18))}
        datasets[key] = {'rows': rows, 'cols': cols, 'weights': weights, 'A_OLS': {'slope': a_slope, 'p': a_p, 'state': a_state}, 'D_Sen_MK_BH': {'slope': slope, 'p': p, 'adjusted_p': adjusted, 'state': d_state}}
        np.savez_compressed(c.OUT / 'vectors' / f'{key}.npz', rows=rows, cols=cols, A_slope=a_slope, A_p=a_p, A_state=a_state, D_sen_slope=slope, D_MK_p=p, D_BH_p=adjusted, D_state=d_state, MK_S=mk_s, MK_variance=variance)
        for suffix, vector, kind in [('sen_slope', slope, 'slope'), ('MK_p', p, 'p'), ('BH_adjusted_p', adjusted, 'p'), ('D_state', d_state, 'state')]:
            write_map(c.OUT / 'maps' / f'{key}_{suffix}.tif', vector, rows, cols, kind, {'VARIABLE': variable, 'PERIOD': f'{years[0]}-{years[-1]}', 'METHOD': 'Theil-Sen slope; tie-corrected continuity-corrected two-sided MK; BH-FDR', 'BH_FAMILY': key, 'BH_FAMILY_SIZE': n, 'ALPHA': c.ALPHA, 'STATE_CODES': '0=decrease,1=not classified significant,2=increase', 'SERIAL_CORRELATION_CORRECTION': 'none'})
        numeric_checks.append({'dataset': key, 'OLS_SciPy_samples': len(indices), 'Sen_SciPy_samples': len(indices), 'BH_statsmodels_full_vector_verified': True})
        print(key, stats_by_method, flush=True)
    long_slope, long_valid = raster(c.LONG_ROOT / 'DI_trend_2001_2021.tif')
    long_p, pvalid = raster(c.LONG_ROOT / 'DI_significance_pvalue_2001_2021.tif')
    long_mask = long_valid & pvalid
    assert int(long_mask.sum()) == c.EXPECTED_LONG
    lr, lc = np.where(long_mask)
    latitude = transform.f + (lr + 0.5) * transform.e
    process('DSI_2001_2021', 'DSI', c.YEARS, series('with_NPV', c.YEARS, long_mask), lr, lc, np.cos(np.deg2rad(latitude)), 'cos(latitude); original Fig.7', long_slope[long_mask].astype(float), long_p[long_mask].astype(float))
    main_statistics = read_json(PROJECT_ROOT / '03_analysis/07_sensitivity/data/weights/all_statistics.json')['entropy_G15']
    baseline_long = statistics['DSI_2001_2021']['methods']['A_OLS']
    np.testing.assert_allclose([baseline_long['decrease_area_percent'], baseline_long['increase_area_percent']], [main_statistics['long_term']['recovery_area_percent'], main_statistics['long_term']['degradation_area_percent']], atol=1e-10)
    _, short_mask = raster(c.ABLATION_ROOT / 'data' / 'figure8' / 'DSI_mean_with_NPV_2004_2019.tif')
    assert int(short_mask.sum()) == c.EXPECTED_SHORT
    sr, sc = np.where(short_mask)
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    row_areas = []
    for row in range(shape[0]):
        north = transform.f + row * transform.e
        south = north + transform.e
        area, _ = geod.polygon_area_perimeter([transform.c, transform.c + transform.a, transform.c + transform.a, transform.c], [north, north, south, south])
        row_areas.append(abs(area) / 1000000.0)
    short_area = np.asarray(row_areas)[sr]
    for variant in ['without_NPV', 'with_NPV']:
        values = series(variant, c.SHORT_YEARS, short_mask)
        process(f'DSI_2004_2019_{variant}', 'DSI ' + variant, c.SHORT_YEARS, values, sr, sc, short_area, 'WGS84 ellipsoidal pixel areas; original Figs.8-9')
        del values
    fvc_root = c.PATHWAY_ROOT / 'data'
    fvc_path = fvc_root / 'fvc_pixel_statistics.npz'
    record(fvc_path)
    fvc = np.load(fvc_path)
    cover_path = fvc_root / 'fvc_annual_cover_pp.npy'
    record(cover_path)
    cover = np.load(cover_path, mmap_mode='r')
    fr, fc = (fvc['rows'], fvc['cols'])
    farea = fvc['area_km2']
    assert len(fr) == c.EXPECTED_FVC and cover.shape == (len(fr), 21, 3)
    for index, name in enumerate(c.COMPONENTS):
        process(f'{name}_2001_2021', name, c.YEARS, cover[:, :, index].T, fr, fc, farea, 'WGS84 ellipsoidal pixel areas; original Fig.10', fvc['trend_pp_per_year'][:, index], fvc['trend_pvalue'][:, index])
    a_labels = 9 * datasets['PV_2001_2021']['A_OLS']['state'] + 3 * datasets['NPV_2001_2021']['A_OLS']['state'] + datasets['BS_2001_2021']['A_OLS']['state'] + 1
    np.testing.assert_array_equal(a_labels, fvc['combination'])
    long_position = np.full(shape, -1, dtype=np.int32)
    long_position[lr, lc] = np.arange(len(lr))
    positions = long_position[fr, fc]
    joint = positions >= 0
    assert int(joint.sum()) == c.EXPECTED_JOINT
    jarea = farea[joint]
    jpositions = positions[joint]
    pathway_rows = []
    pathway_results = {}
    pathway_source = read_json(fvc_root / 'main_dsi_response_summary.json')
    for method in c.METHODS:
        component_states = np.column_stack([datasets[f'{name}_2001_2021'][method]['state'] for name in c.COMPONENTS])
        labels = (9 * component_states[:, 0] + 3 * component_states[:, 1] + component_states[:, 2] + 1).astype(np.uint8)
        paired_labels = labels[joint]
        slope = datasets['DSI_2001_2021'][method]['slope'][jpositions]
        dsi_state = datasets['DSI_2001_2021'][method]['state'][jpositions]
        classes = []
        for code in range(1, 28):
            selected = paired_labels == code
            all_selected = labels == code
            selected_area = jarea[selected]
            rec = {'method': method, 'code': code, 'PV_state': (code - 1) // 9, 'NPV_state': (code - 1) % 9 // 3, 'BS_state': (code - 1) % 3, 'full_FVC_pixels': int(all_selected.sum()), 'full_FVC_area_percent': float(farea[all_selected].sum() / farea.sum() * 100), 'paired_pixels': int(selected.sum()), 'paired_area_percent': float(selected_area.sum() / jarea.sum() * 100), 'median_DSI_slope_per_year': None, 'DSI_increase_within_group_percent': None, 'significant_DSI_increase_within_group_percent': None}
            if selected.any():
                rec.update(median_DSI_slope_per_year=float(weighted_quantile(slope[selected], selected_area)[0]), DSI_increase_within_group_percent=float(selected_area[slope[selected] > 0].sum() / selected_area.sum() * 100), significant_DSI_increase_within_group_percent=float(selected_area[dsi_state[selected] == 2].sum() / selected_area.sum() * 100))
            if method == 'A_OLS':
                old = pathway_source['classes'][code - 1]
                np.testing.assert_allclose(rec['paired_area_percent'], old['paired_area_percent'], atol=1e-09)
                if selected.any():
                    np.testing.assert_allclose(rec['median_DSI_slope_per_year'], old['dsi_trend_quantiles_per_year'][2], atol=1e-12)
                    np.testing.assert_allclose(rec['DSI_increase_within_group_percent'], old['dsi_direction_percent_decrease_increase'][1], atol=1e-09)
            classes.append(rec)
            pathway_rows.append(rec)
        pathway_results[method] = classes
        write_map(c.OUT / 'maps' / f'FVC_pathways_{method}.tif', labels, fr, fc, 'state', {'METHOD': method, 'CODES': '1..27=9*sPV+3*sNPV+sBS+1; states decrease=0,NS=1,increase=2', 'FVC_DOMAIN': len(fr), 'BH_FAMILIES': 'each component full FVC domain separately'})
    save_csv(c.OUT / 'tables' / 'pathways_all27_A_D.csv', pathway_rows)
    external_y = []
    external_class = []
    external_valid = np.ones(len(sr), dtype=bool)
    product_meta = []
    for product in c.PRODUCTS:
        values, valid = raster(c.external_path(product))
        minimum, maximum = (float(values[valid].min()), float(values[valid].max()))
        lower, upper = (minimum * 0.1, maximum * 0.1)
        values = values[short_mask]
        current = np.ones(len(sr), dtype=np.uint8)
        current[values < lower] = 0
        current[values > upper] = 2
        external_y.append(-values.astype(float) / maximum)
        external_class.append(current)
        external_valid &= valid[short_mask]
        product_meta.append({'product': product, 'lower_threshold': lower, 'upper_threshold': upper, 'maximum_normalizer': maximum})
    assert int(external_valid.sum()) == c.EXPECTED_EXTERNAL
    external_y = np.asarray(external_y)[:, external_valid]
    external_class = np.asarray(external_class)[:, external_valid]
    longitude = transform.c + (sc[external_valid] + 0.5) * transform.a
    latitude = transform.f + (sr[external_valid] + 0.5) * transform.e
    bx = np.floor((longitude + 180) / c.BLOCK_SIZE_DEG).astype(int)
    by = np.floor((latitude + 90) / c.BLOCK_SIZE_DEG).astype(int)
    block_labels = by * (int(np.ceil(360 / c.BLOCK_SIZE_DEG)) + 1) + bx
    blocks, block_index = np.unique(block_labels, return_inverse=True)
    assert len(blocks) == 411
    contrast = {}
    external = {}
    external_rows = []
    contrast_rows = []
    for method in c.METHODS:
        pair = [datasets[f'DSI_2004_2019_{variant}'][method] for variant in ['without_NPV', 'with_NPV']]
        pair_states = np.array([item['state'] for item in pair])
        slope = np.array([item['slope'] for item in pair])
        changed = pair_states[0] != pair_states[1]
        reversal = (pair_states[0] == 0) & (pair_states[1] == 2) | (pair_states[0] == 2) & (pair_states[1] == 0)
        matrix = np.bincount(pair_states[0] * 3 + pair_states[1], weights=short_area, minlength=9).reshape(3, 3) / short_area.sum() * 100
        result = {'method': method, 'pixels': len(sr), 'mean_delta_DSI': main_statistics['NPV_contrast']['mean_delta_DSI'], 'changed_class_area_percent': float(short_area[changed].sum() / short_area.sum() * 100), 'direct_reversal_area_percent': float(short_area[reversal].sum() / short_area.sum() * 100), 'transition_matrix_decrease_NS_increase_percent': matrix.tolist()}
        for index, variant in enumerate(['without_NPV', 'with_NPV']):
            baseline = datasets[f'DSI_2004_2019_{variant}']['A_OLS']['state']
            result[f'{variant}_class_agreement_with_A_area_percent'] = float(short_area[pair_states[index] == baseline].sum() / short_area.sum() * 100)
        contrast[method] = result
        contrast_rows.append({key: value for key, value in result.items() if not isinstance(value, list)})
        if method == 'A_OLS':
            np.testing.assert_allclose(result['changed_class_area_percent'], main_statistics['NPV_contrast']['changed_trend_class_area_percent'], atol=1e-09)
            np.testing.assert_allclose(matrix[::-1, ::-1], main_statistics['transition_matrix_area_percent'], atol=1e-09)
        print(f'External bootstrap: {method}', flush=True)
        metrics, dr, dba = paired_external(slope[:, external_valid], 2 - pair_states[:, external_valid], external_y, external_class, block_index, len(blocks), reps=c.BOOTSTRAP_REPLICATES, batch=c.BOOTSTRAP_BATCH, seed=c.SEED + round(c.BLOCK_SIZE_DEG * 100))
        external[method] = metrics
        np.savez_compressed(c.OUT / 'vectors' / f'paired_bootstrap_{method}.npz', delta_r=dr, delta_BA_pp=dba)
        if method == 'A_OLS':
            for key in ['r', 'ba_percent', 'delta_r', 'delta_ba_pp', 'delta_r_ci95', 'delta_ba_pp_ci95']:
                np.testing.assert_allclose(metrics[key], main_statistics['external'][key], atol=1e-11)
        for index, product in enumerate(c.PRODUCTS):
            external_rows.append({'method': method, 'product': product, 'r_without_NPV': metrics['r'][index][0], 'r_with_NPV': metrics['r'][index][1], 'delta_r': metrics['delta_r'][index], 'delta_r_CI_low': metrics['delta_r_ci95'][index][0], 'delta_r_CI_high': metrics['delta_r_ci95'][index][1], 'BA_without_NPV_percent': metrics['ba_percent'][index][0], 'BA_with_NPV_percent': metrics['ba_percent'][index][1], 'delta_BA_pp': metrics['delta_ba_pp'][index], 'delta_BA_CI_low_pp': metrics['delta_ba_pp_ci95'][index][0], 'delta_BA_CI_high_pp': metrics['delta_ba_pp_ci95'][index][1]})
    save_csv(c.OUT / 'tables' / 'NPV_contrasts_A_D.csv', contrast_rows)
    save_csv(c.OUT / 'tables' / 'external_consistency_A_D.csv', external_rows)
    rows = []
    for dataset, summary in statistics.items():
        for method, metrics in summary['methods'].items():
            rows.append({'dataset': dataset, 'method': method, 'pixels': summary['pixels'], **metrics})
    save_csv(c.OUT / 'tables' / 'trend_classification_A_D.csv', rows)
    results = {'methods': {'A_OLS': 'OLS slope; original nominal two-sided Student t P<0.05', 'D_Sen_MK_BH': 'Theil-Sen slope; tie-corrected, continuity-corrected two-sided MK; BH adjusted P<0.05'}, 'datasets': statistics, 'contrast': contrast, 'external': external, 'pathways': pathway_results, 'display_pathways': list(c.MAIN_PATHWAYS), 'domains': {'long_DSI': len(lr), 'paired_DSI': len(sr), 'external': int(external_valid.sum()), 'FVC': len(fr), 'joint_FVC_DSI': int(joint.sum())}, 'bootstrap': {'blocks': len(blocks), 'block_degrees': c.BLOCK_SIZE_DEG, 'replicates': c.BOOTSTRAP_REPLICATES, 'paired_seed': c.SEED + round(c.BLOCK_SIZE_DEG * 100)}, 'external_product_metadata': product_meta, 'numeric_checks': numeric_checks, 'scope': 'A and D only; differences combine estimator, test and multiplicity adjustment; no serial-correlation correction'}
    save_json(c.OUT / 'statistics.json', results)
    original_codes = [CODE_ROOT / '03_analysis/05_cover_combinations/code/07_compute_fvc_combinations.py', CODE_ROOT / '03_analysis/05_cover_combinations/code/08_compute_main_dsi_response.py', CODE_ROOT / '03_analysis/04_npv_ablation/code/02_compute_figure9_npv_trend_effect.py', CODE_ROOT / '04_plotting/code/spacetime/04_plot_DI_trend_and_significance.py']
    for path in original_codes:
        record(path)
    for item in sources.values():
        verify_source_record(item)
    save_json(c.OUT / 'audit' / 'complete.json', {'utc_time': datetime.now(timezone.utc).isoformat(), 'code_signature': signature(), 'sources': list(sources.values()), 'baseline_trends_Fig7_reproduced': True, 'baseline_NPV_Fig9_and_paired_intervals_reproduced': True, 'baseline_all27_Fig10_statistics_reproduced': True, 'numeric_checks': numeric_checks})
    print('A_D_ANALYSIS_COMPLETE', str(c.OUT), flush=True)
if __name__ == '__main__':
    main()
