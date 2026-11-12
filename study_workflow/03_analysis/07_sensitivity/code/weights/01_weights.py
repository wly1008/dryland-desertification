"""Recount G=10/15/20 on unchanged annual masks; verify baseline weights."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import hashlib
import platform
import sys
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import config as c
from common import entropy_weights, file_record, finish_stage, write_csv, write_json

def main():
    gs = (10, 15, 20)
    pooled = {g: np.zeros((len(c.INDICATORS), g), dtype=np.int64) for g in gs}
    annual_rows, manifest, validity_rows = ([], [], [])
    count_path = c.COMMON_ROOT / '05class_statistics' / 'class_counts.xlsx'
    stored_counts = pd.read_excel(count_path, sheet_name=None, index_col=0)
    manifest.append(file_record(count_path))
    for year in tqdm(c.YEARS, desc='Count normalized indicators on fixed masks', unit='year'):
        with rasterio.open(c.mask_path(year)) as source:
            mask = source.read(1) == 1
            shape, crs, transform = (source.shape, source.crs, source.transform)
            tags = source.tags()
        if tags.get('PREPROCESS_INDICATORS') != ','.join(c.ALL_PREPROCESS_INDICATORS):
            raise ValueError(f'Unexpected original preprocessing indicators: {year}')
        manifest.append(file_record(c.mask_path(year)))
        n = int(mask.sum())
        validity_rows.append({'year': year, 'common_valid_pixels': n, 'mask_indicators': ','.join(c.ALL_PREPROCESS_INDICATORS)})
        for index, indicator in enumerate(c.INDICATORS):
            path = c.input_path(indicator, year)
            with rasterio.open(path) as source:
                if source.shape != shape or source.crs != crs or (not source.transform.almost_equals(transform)):
                    raise ValueError(f'Grid mismatch: {path}')
                raw = source.read(1, masked=True)
                valid = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data) & (raw.data != c.INPUT_NODATA)
                if np.any(mask & ~valid):
                    raise ValueError(f'Missing normalized input inside fixed mask: {path}')
                if source.tags().get('STATISTIC') != 'mean':
                    raise ValueError(f'Input is not mean: {path}')
            manifest.append(file_record(path))
            normalized = np.clip(raw.data[mask].astype(np.float64) / c.INPUT_SCALE_MAX, 0, 1)
            for g in gs:
                classes = np.minimum(np.floor(normalized * g).astype(np.int16), g - 1)
                counts = np.bincount(classes, minlength=g)
                if g == 15:
                    expected = stored_counts[str(year)][indicator].iloc[:g].to_numpy(dtype=np.int64)
                    np.testing.assert_array_equal(counts, expected, err_msg=f'Baseline histogram {indicator} {year}')
                pooled[g][index] += counts
                annual_rows.extend(({'year': year, 'indicator': indicator, 'G': g, 'class': group + 1, 'count': int(value)} for group, value in enumerate(counts)))
    weight_config, weight_rows = ({}, [])
    for scenario, g in c.SCENARIOS.items():
        if g is None:
            entropy = np.full(7, np.nan)
            with_weights = np.full(7, 1 / 7)
        else:
            entropy, with_weights = entropy_weights(pooled[g])
        without_weights = with_weights[:-1] / with_weights[:-1].sum()
        if scenario == c.BASELINE:
            for variant, calculated in [('with_NPV', with_weights), ('without_NPV', without_weights)]:
                path = c.WEIGHT_ROOT / c.group_name(variant) / '06Gk_Er_W' / 'Gk_Er_W.xlsx'
                stored = pd.read_excel(path, sheet_name='W', index_col=0)['study_period']
                names = c.INDICATORS if variant == 'with_NPV' else c.WITHOUT_NPV
                actual = stored.loc[list(names)].to_numpy(dtype=np.float64)
                np.testing.assert_allclose(actual, calculated, rtol=0, atol=5e-13)
                if variant == 'with_NPV':
                    with_weights = actual
                else:
                    without_weights = actual
                manifest.append(file_record(path))
        np.testing.assert_allclose(without_weights, with_weights[:-1] / (1 - with_weights[-1]), atol=5e-13)
        weight_config[scenario] = {'G': g, 'method': 'equal' if g is None else 'entropy', 'with_NPV': dict(zip(c.INDICATORS, with_weights.tolist())), 'without_NPV': dict(zip(c.WITHOUT_NPV, without_weights.tolist()))}
        for variant, names, weights in [('with_NPV', c.INDICATORS, with_weights), ('without_NPV', c.WITHOUT_NPV, without_weights)]:
            for i, (name, weight) in enumerate(zip(names, weights)):
                weight_rows.append({'scenario': scenario, 'variant': variant, 'G': g if g is not None else 'NA', 'indicator': name, 'entropy_bits': float(entropy[i]) if g is not None else 'NA', 'weight': float(weight)})
    out = c.OUTPUT_ROOT / 'weights'
    write_json(out / 'weights.json', weight_config)
    write_csv(out / 'annual_class_counts.csv', annual_rows)
    write_csv(out / 'weights_long.csv', weight_rows)
    write_csv(out / 'annual_validity.csv', validity_rows)
    write_json(c.OUTPUT_ROOT / 'audit' / 'source_manifest.json', manifest)
    settings = {'output_root': str(c.OUTPUT_ROOT), 'years': list(c.YEARS), 'ablation_years': list(c.ABLATION_YEARS), 'scenarios': c.SCENARIOS, 'indicators': list(c.INDICATORS), 'fixed_annual_mask_indicators': list(c.ALL_PREPROCESS_INDICATORS), 'weight_estimation': 'pixel-season equal counts, pooled 2001-2021; fixed study-period weights', 'direction': 'already normalized: higher means greater desertification', 'long_area_weighting': 'cos(latitude), original Figure 7 domain', 'ablation_area_weighting': 'WGS84 ellipsoidal pixel area, original Figure 8/9 domain', 'external_weighting': 'equal pixel weights, original Figure 9 common domain', 'bootstrap': {'block_degrees': c.BLOCK_SIZE_DEG, 'replicates': c.BOOTSTRAP_REPLICATES, 'seed': c.RANDOM_SEED}, 'python': sys.version, 'platform': platform.platform(), 'original_config_sha256': hashlib.sha256((c.ORIGINAL_CODE_ROOT / 'config.py').read_bytes()).hexdigest()}
    write_json(c.OUTPUT_ROOT / 'config_snapshot.json', settings)
    finish_stage('01_weights', pooled_pixel_seasons=sum((x['common_valid_pixels'] for x in validity_rows)), baseline_histograms_exact=True, baseline_weight_tolerance=5e-13)
    print('Weights ready; original G15 histograms and full-precision weights verified.', flush=True)
if __name__ == '__main__':
    main()
