"""Numerical and file helpers shared by the sensitivity stages."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path, source_record
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import rasterio
from scipy import stats
from pyproj import Geod
import config as c

def write_json(path, value):
    path = remap_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def read_json(path):
    return json.loads(remap_path(path).read_text(encoding='utf-8'))

def write_csv(path, rows):
    rows = list(rows)
    path = remap_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def file_record(path):
    path = remap_path(path)
    return source_record(path)

def code_signature():
    digest = hashlib.sha256()
    for path in sorted(c.CODE_ROOT.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def finish_stage(stage, **information):
    write_json(c.stage_path(stage), {'utc_time': datetime.now(timezone.utc).isoformat(), 'code_signature': code_signature(), **information})

def assert_grid(source, reference):
    if source.shape != reference['shape'] or source.crs != reference['crs'] or (not source.transform.almost_equals(reference['transform'])):
        raise ValueError(f'Grid mismatch: {source.name}')

def valid_array(source):
    values = source.read(1, masked=True)
    return (values.data, ~np.ma.getmaskarray(values) & np.isfinite(values.data))

def entropy_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    if np.any(totals <= 0) or not np.all(totals == totals[0]):
        raise ValueError('Entropy indicators must share the same positive sample size')
    probabilities = counts / totals[:, None]
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])
    entropy = terms.sum(axis=1)
    if entropy.sum() <= 0:
        raise ValueError('All entropies are zero')
    return (entropy, entropy / entropy.sum())

def ols(values, years):
    """OLS with a fresh residual-based two-sided t-test for each series."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(years, dtype=np.float64)
    times -= times.mean()
    xss = np.sum(times ** 2)
    centered = values - values.mean(axis=0)
    slope = np.sum(times[:, None] * centered, axis=0) / xss
    residuals = centered - times[:, None] * slope
    se = np.sqrt(np.sum(residuals ** 2, axis=0) / (len(years) - 2) / xss)
    t = np.divide(np.abs(slope), se, out=np.zeros_like(slope), where=se > 0)
    p = 2 * stats.t.sf(t, len(years) - 2)
    p[(se == 0) & (slope != 0)] = 0.0
    return (slope, p)

def trend_classes(slope, p):
    classes = np.ones(len(slope), dtype=np.uint8)
    classes[(slope > 0) & (p < c.ALPHA)] = 0
    classes[(slope < 0) & (p < c.ALPHA)] = 2
    return classes

def area_percentages(classes, weights):
    return np.bincount(classes, weights=weights, minlength=3) / weights.sum() * 100

def cell_areas(shape, transform):
    geod = Geod(ellps='WGS84')
    areas = np.empty(shape[0])
    west, east = (transform.c, transform.c + transform.a)
    for row in range(shape[0]):
        north = transform.f + row * transform.e
        south = north + transform.e
        signed, _ = geod.polygon_area_perimeter([west, east, east, west], [north, north, south, south])
        areas[row] = abs(signed) / 1000000.0
    return areas

def write_raster(path, values, mask, profile, tags=None, categorical=False):
    path = remap_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype, nodata = ('uint8', 255) if categorical else ('float32', c.ANALYSIS_NODATA)
    array = np.full(mask.shape, nodata, dtype=dtype)
    array[mask] = values
    options = profile.copy()
    options.pop('nbits', None)
    options.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress='deflate', tiled=True, blockxsize=256, blockysize=256, predictor=2 if categorical else 3)
    temporary = path.with_name(path.stem + '.part.tif')
    with rasterio.open(temporary, 'w', **options) as destination:
        destination.write(array, 1)
        destination.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    with rasterio.open(temporary) as source:
        check = source.read(1)
        if not np.array_equal(check[mask], array[mask]) or not np.all(check[~mask] == nodata):
            raise AssertionError(f'Output raster verification failed: {path}')
    temporary.replace(path)

def paired_external(slopes, classes, external_y, reference_classes, block_index, n_blocks, *, save_draws=None):
    """Unweighted pixel metrics; exact block sufficient statistics, paired draws."""
    n = slopes.shape[1]
    sufficient = np.zeros((n_blocks, 3, 27), dtype=np.float64)
    x = (slopes - slopes.mean(axis=1, keepdims=True)) / slopes.std(axis=1, keepdims=True)
    y = (external_y - external_y.mean(axis=1, keepdims=True)) / external_y.std(axis=1, keepdims=True)
    counts = np.bincount(block_index, minlength=n_blocks)
    for product in range(3):
        sufficient[:, product, 0] = counts
        terms = [x[0], x[1], y[product], x[0] ** 2, x[1] ** 2, y[product] ** 2, x[0] * y[product], x[1] * y[product]]
        for column, term in enumerate(terms, 1):
            sufficient[:, product, column] = np.bincount(block_index, weights=term, minlength=n_blocks)
        for variant in range(2):
            cells = reference_classes[product] * 3 + classes[variant]
            sufficient[:, product, 9 + variant * 9:18 + variant * 9] = np.bincount(block_index * 9 + cells, minlength=n_blocks * 9).reshape(n_blocks, 9)

    def calculate(totals):
        counts = totals[..., 0]
        rs, bas = ([], [])
        for variant in range(2):
            numerator = totals[..., 7 + variant] - totals[..., 1 + variant] * totals[..., 3] / counts
            xss = totals[..., 4 + variant] - totals[..., 1 + variant] ** 2 / counts
            yss = totals[..., 6] - totals[..., 3] ** 2 / counts
            rs.append(numerator / np.sqrt(xss * yss))
            confusion = totals[..., 9 + variant * 9:18 + variant * 9].reshape(*totals.shape[:-1], 3, 3)
            denominators = confusion.sum(axis=-1)
            if np.any(denominators == 0):
                raise ValueError('A bootstrap replicate has an absent reference class')
            bas.append((np.diagonal(confusion, axis1=-2, axis2=-1) / denominators).mean(axis=-1))
        return (np.stack(rs, axis=-1), np.stack(bas, axis=-1))
    r, ba = calculate(sufficient.sum(axis=0))
    for product in range(3):
        for variant in range(2):
            np.testing.assert_allclose(r[product, variant], stats.pearsonr(slopes[variant], external_y[product]).statistic, atol=1e-12)
            cm = np.bincount(reference_classes[product] * 3 + classes[variant], minlength=9).reshape(3, 3)
            np.testing.assert_allclose(ba[product, variant], (np.diag(cm) / cm.sum(axis=1)).mean(), atol=1e-12)
    rng = np.random.default_rng(c.RANDOM_SEED + round(c.BLOCK_SIZE_DEG * 100))
    r_draws = np.empty((c.BOOTSTRAP_REPLICATES, 3, 2))
    ba_draws = np.empty_like(r_draws)
    for start in range(0, c.BOOTSTRAP_REPLICATES, c.BOOTSTRAP_BATCH):
        stop = min(start + c.BOOTSTRAP_BATCH, c.BOOTSTRAP_REPLICATES)
        multiplicities = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=stop - start)
        totals = (multiplicities @ sufficient.reshape(n_blocks, -1)).reshape(stop - start, 3, 27)
        r_draws[start:stop], ba_draws[start:stop] = calculate(totals)
    dr = r_draws[:, :, 1] - r_draws[:, :, 0]
    dba = (ba_draws[:, :, 1] - ba_draws[:, :, 0]) * 100
    r_ci, ba_ci = (np.quantile(dr, [0.025, 0.975], axis=0).T, np.quantile(dba, [0.025, 0.975], axis=0).T)
    if save_draws is not None:
        np.savez_compressed(save_draws, delta_r=dr, delta_ba_pp=dba)
    return {'pixels': n, 'blocks': n_blocks, 'r': r.tolist(), 'ba_percent': (ba * 100).tolist(), 'delta_r': (r[:, 1] - r[:, 0]).tolist(), 'delta_ba_pp': ((ba[:, 1] - ba[:, 0]) * 100).tolist(), 'delta_r_ci95': r_ci.tolist(), 'delta_ba_pp_ci95': ba_ci.tolist()}
