"""Calculate pixel-wise OLS slopes, two-sided P values, and significant slopes."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import rasterio
from scipy import stats
ROOT = next((p for p in Path(__file__).resolve().parents if (p / '07_environment').is_dir()))
sys.path.insert(0, str(ROOT / '07_environment'))
from project_paths import project_relative, PROJECT_ROOT
ROOT = PROJECT_ROOT

def compute(source, years):
    valid = None
    arrays = []
    reference = None
    for year in years:
        path = source / f'DI_{year}.tif'
        with rasterio.open(path) as src:
            grid = (src.shape, src.crs, src.transform)
            if reference is None:
                reference = grid
                profile = src.profile.copy()
            elif grid != reference:
                raise ValueError('Annual grids differ: ' + str(path))
            arr = src.read(1, masked=True)
            good = ~np.ma.getmaskarray(arr) & np.isfinite(arr.data)
            valid = good.copy() if valid is None else valid & good
            arrays.append(arr.data)
    if not valid.any():
        raise ValueError('No complete annual series')
    values = np.stack([x[valid] for x in arrays]).astype(np.float64)
    t = np.asarray(years, dtype=float)
    t -= t.mean()
    sxx = np.sum(t * t)
    centered = values - values.mean(axis=0)
    slope = np.sum(t[:, None] * centered, axis=0) / sxx
    residual = centered - t[:, None] * slope
    se = np.sqrt(np.sum(residual * residual, axis=0) / (len(years) - 2) / sxx)
    tv = np.divide(np.abs(slope), se, out=np.zeros_like(slope), where=se > 0)
    pvalue = 2 * stats.t.sf(tv, len(years) - 2)
    pvalue[(se == 0) & (slope != 0)] = 0
    for index in np.linspace(0, len(slope) - 1, min(32, len(slope)), dtype=int):
        fit = stats.linregress(years, values[:, index])
        np.testing.assert_allclose(slope[index], fit.slope, atol=1e-14, rtol=1e-10)
        np.testing.assert_allclose(pvalue[index], fit.pvalue, atol=1e-11, rtol=1e-08)
    return (valid, profile, slope, pvalue)

def export(path, data, valid, profile, tags, overwrite, tolerance):
    if path.exists() and (not overwrite):
        with rasterio.open(path) as old:
            if old.shape != valid.shape or old.transform != profile['transform'] or old.crs != profile['crs']:
                raise ValueError('Existing output grid differs: ' + str(path))
            arr = old.read(1, masked=True)
            mask = ~np.ma.getmaskarray(arr)
            np.testing.assert_array_equal(mask, valid, err_msg=str(path))
            np.testing.assert_allclose(arr.data[valid], data, atol=tolerance, rtol=0, err_msg=str(path))
        return 'verified existing'
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full(valid.shape, 32767, np.float32)
    array[valid] = data
    options = profile.copy()
    options.update(driver='GTiff', dtype='float32', nodata=32767, count=1, compress='deflate')
    temporary = path.with_name(path.stem + '.part.tif')
    try:
        with rasterio.open(temporary, 'w', **options) as dst:
            dst.write(array, 1)
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return 'written'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=int, default=2001)
    p.add_argument('--end', type=int, default=2021)
    p.add_argument('--variant', choices=['both', 'with_NPV', 'without_NPV'], default='with_NPV')
    p.add_argument('--all-periods', action='store_true')
    p.add_argument('--publish', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if not 2001 <= a.start < a.end <= 2021:
        p.error('Years must lie within 2001–2021')
    if a.out and (a.variant == 'both' or a.all_periods):
        p.error('--out requires one variant and one period')
    reports = []
    for variant in ['without_NPV', 'with_NPV'] if a.variant == 'both' else [a.variant]:
        group = 'Albedo_SMI_TGSI_Wet_LST_PV' + ('_NPV' if variant == 'with_NPV' else '')
        source = ROOT / '03_analysis/01_dsi/data/02DI' / group / '02study_period_weights'
        for start, end in [(2001, 2021), (2004, 2019)] if a.all_periods else [(a.start, a.end)]:
            years = np.arange(start, end + 1)
            valid, profile, slope, pvalue = compute(source, years)
            base = ROOT / '03_analysis/02_trends/data' / group / '02study_period_weights'
            out = a.out or (base if a.publish else ROOT / f'06_verification/recomputed_trends/{variant}_{start}_{end}')
            common = {'START_YEAR': start, 'END_YEAR': end, 'YEAR_COUNT': len(years), 'VALIDITY_RULE': 'all selected years must be valid', 'VALID_PIXEL_COUNT': int(valid.sum())}
            specs = [(f'DI_trend_{start}_{end}.tif', slope, valid, 'pixel-wise ordinary least squares linear trend', 1e-08), (f'DI_significance_pvalue_{start}_{end}.tif', pvalue, valid, 'two-sided Student t-test for OLS slope', 1e-06)]
            sig = np.zeros(valid.shape, bool)
            sig[valid] = pvalue < 0.05
            specs.append((f'DI_trend_p_lt_0.05_{start}_{end}.tif', slope[pvalue < 0.05], sig, 'trend retained where two-sided p-value is below threshold', 1e-08))
            record = {'variant': variant, 'start': start, 'end': end, 'pixels': int(valid.sum()), 'outputs': {}}
            for name, values, mask, analysis, tolerance in specs:
                record['outputs'][name] = export(out / name, values, mask, profile, {**common, 'ANALYSIS': analysis, 'SOURCE_DIRECTORY': project_relative(source), 'PVALUE_THRESHOLD': 0.05}, a.overwrite, tolerance)
                if not a.publish and (base / name).exists():
                    export(base / name, values, mask, profile, {}, False, tolerance)
            out.mkdir(parents=True, exist_ok=True)
            if not a.publish:
                (out / 'comparison.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            reports.append(record)
            print(json.dumps(record), flush=True)
    log = ROOT / '06_verification/trends_latest_run.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(reports, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
