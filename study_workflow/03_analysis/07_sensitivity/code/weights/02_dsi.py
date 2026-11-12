"""Verify original G15 annual DSI and write six alternative annual series."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import numpy as np
import rasterio
from tqdm import tqdm
import config as c
from common import assert_grid, file_record, finish_stage, read_json, valid_array, write_json, write_raster

def main():
    weights = read_json(c.OUTPUT_ROOT / 'weights' / 'weights.json')
    errors, records = ([], [])
    for year in tqdm(c.YEARS, desc='Build alternative annual DSI and verify baseline', unit='year'):
        with rasterio.open(c.mask_path(year)) as source:
            mask = source.read(1) == 1
            profile = source.profile.copy()
            reference = {'shape': source.shape, 'crs': source.crs, 'transform': source.transform}
        z = {}
        for indicator in c.INDICATORS:
            with rasterio.open(c.input_path(indicator, year)) as source:
                assert_grid(source, reference)
                raw, valid = valid_array(source)
                if np.any(mask & ~valid):
                    raise ValueError('Input missing inside original mask')
            z[indicator] = np.clip(raw[mask].astype(np.float64) / c.INPUT_SCALE_MAX, 0, 1)
        for scenario in c.SCENARIOS:
            for variant in c.VARIANTS:
                annual = np.zeros(int(mask.sum()), dtype=np.float64)
                for indicator, weight in weights[scenario][variant].items():
                    annual += weight * z[indicator]
                annual = np.clip(annual, 0, 1).astype(np.float32)
                if scenario == c.BASELINE:
                    path = c.original_dsi_path(variant, year)
                    with rasterio.open(path) as source:
                        assert_grid(source, reference)
                        original, valid = valid_array(source)
                        if not np.array_equal(valid, mask):
                            raise ValueError(f'Baseline mask mismatch: {path}')
                        if source.tags().get('WEIGHT_MODE') != 'study_period_weights':
                            raise ValueError(f'Baseline not fixed weights: {path}')
                    error = float(np.max(np.abs(original[mask].astype(float) - annual)))
                    np.testing.assert_allclose(original[mask], annual, rtol=0, atol=6e-08)
                    errors.append({'year': year, 'variant': variant, 'max_absolute_error': error})
                else:
                    path = c.dsi_path(scenario, variant, year)
                    tags = {'SCENARIO': scenario, 'WEIGHT_METHOD': weights[scenario]['method'], 'BIN_COUNT': weights[scenario]['G'], 'WEIGHT_MODE': 'study_period_weights', 'YEAR': year, 'PERIOD_START': f'{year}-10', 'PERIOD_END': f'{year + 1}-04', 'STATISTIC': 'mean', 'INDICATORS': ','.join(weights[scenario][variant]), 'APPLIED_WEIGHTS': ';'.join((f'{k}={v:.17g}' for k, v in weights[scenario][variant].items())), 'DIRECTION_STATUS': 'all indicators already unified; higher means more desertification', 'DI_FORMULA': 'DI=sum(W_k*clip(x_k/10000,0,1))', 'MASK': str(c.mask_path(year)), 'NPV_CONTRAST': 'with NPV minus without NPV; other weights renormalized'}
                    write_raster(path, annual, mask, profile, tags)
                records.append({'scenario': scenario, 'variant': variant, 'year': year, **file_record(path)})
    write_json(c.OUTPUT_ROOT / 'audit' / 'baseline_annual_reproduction.json', errors)
    write_json(c.OUTPUT_ROOT / 'audit' / 'annual_dsi_manifest.json', records)
    finish_stage('02_dsi', baseline_max_absolute_error=max((x['max_absolute_error'] for x in errors)), original_annual_rasters_verified=len(errors), new_annual_rasters_written=6 * len(c.YEARS))
    print('Annual DSI ready; existing baseline has not been modified.', flush=True)
if __name__ == '__main__':
    main()
