"""Numerical kernels. Original Fig.9 pairing and Fig.10 coding are retained."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import numpy as np
from scipy import stats

def ols(values, years):
    times = np.asarray(years, dtype=np.float64)
    times = times - times.mean()
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    sxx = np.sum(times ** 2)
    slope = np.sum(times[:, None] * centered, axis=0) / sxx
    residual = centered - times[:, None] * slope
    se = np.sqrt(np.sum(residual ** 2, axis=0) / (len(years) - 2) / sxx)
    t = np.divide(np.abs(slope), se, out=np.zeros_like(slope), where=se > 0)
    p = 2 * stats.t.sf(t, len(years) - 2)
    p[(se == 0) & (slope != 0)] = 0
    return (slope, p)

def sen_mk(values, years):
    """One chunk, shape time x pixels. Median of all pairwise time slopes.

    Original MK normal approximation, exact-value tie correction and continuity
    correction. Constant series have S=0, variance=0, Z=0 and P=1.
    This function performs no serial-correlation correction/prewhitening.
    """
    values = np.asarray(values, dtype=np.float64)
    n, m = values.shape
    times = np.asarray(years, dtype=np.float64)
    if len(times) != n or np.any(np.diff(times) <= 0) or (not np.isfinite(values).all()):
        raise ValueError('Require increasing times and complete finite series')
    i, j = np.triu_indices(n, 1)
    differences = values[j] - values[i]
    s = np.sign(differences).sum(axis=0).astype(np.int32)
    differences /= (times[j] - times[i])[:, None]
    slope = np.median(differences, axis=0)
    del differences
    ordered = np.sort(values, axis=0)
    run = np.ones(m, dtype=np.int32)
    tie_term = np.zeros(m, dtype=np.float64)
    for t in range(1, n):
        equal = ordered[t] == ordered[t - 1]
        ended = ~equal
        lengths = run[ended].astype(np.float64)
        tie_term[ended] += lengths * (lengths - 1) * (2 * lengths + 5)
        run[equal] += 1
        run[ended] = 1
    lengths = run.astype(np.float64)
    tie_term += lengths * (lengths - 1) * (2 * lengths + 5)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    z = np.divide(s - np.sign(s), np.sqrt(variance), out=np.zeros(m), where=variance > 0)
    p = 2 * stats.norm.sf(np.abs(z))
    return (slope, p, s, variance)

def bh_adjust(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('BH family must be a finite vector of valid p values')
    order = np.argsort(p, kind='stable')
    sorted_adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(sorted_adjusted, 1)
    return result

def states(slope, p, alpha=0.05):
    """0 decreasing, 1 not classified significant, 2 increasing."""
    result = np.ones(len(slope), dtype=np.uint8)
    result[(slope < 0) & (p < alpha)] = 0
    result[(slope > 0) & (p < alpha)] = 2
    return result

def area_fractions(state, weights):
    return np.bincount(state, weights=weights, minlength=3) / weights.sum() * 100

def weighted_quantile(values, weights, probabilities=(0.5,)):
    order = np.argsort(values)
    x = (np.cumsum(weights[order]) - weights[order] / 2) / weights.sum()
    return np.interp(probabilities, x, values[order])

def paired_external(slopes, classes, y, reference_classes, block_index, n_blocks, reps=1999, batch=100, seed=20261413):
    """Replicate the original Fig.9 5-degree paired block bootstrap.

    classes use external ordering: degradation=0, NS=1, recovery=2.
    Weights are equal pixels. Fixed classifications are resampled, not refit.
    """
    sufficient = np.zeros((n_blocks, 3, 27), dtype=np.float64)
    x = (slopes - slopes.mean(axis=1, keepdims=True)) / slopes.std(axis=1, keepdims=True)
    yy = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    counts = np.bincount(block_index, minlength=n_blocks)
    for k in range(3):
        sufficient[:, k, 0] = counts
        terms = [x[0], x[1], yy[k], x[0] ** 2, x[1] ** 2, yy[k] ** 2, x[0] * yy[k], x[1] * yy[k]]
        for column, term in enumerate(terms, 1):
            sufficient[:, k, column] = np.bincount(block_index, weights=term, minlength=n_blocks)
        for variant in range(2):
            cells = reference_classes[k] * 3 + classes[variant]
            sufficient[:, k, 9 + variant * 9:18 + variant * 9] = np.bincount(block_index * 9 + cells, minlength=n_blocks * 9).reshape(n_blocks, 9)

    def metrics(totals):
        n = totals[..., 0]
        rs, bas = ([], [])
        for v in range(2):
            numerator = totals[..., 7 + v] - totals[..., 1 + v] * totals[..., 3] / n
            xss = totals[..., 4 + v] - totals[..., 1 + v] ** 2 / n
            yss = totals[..., 6] - totals[..., 3] ** 2 / n
            rs.append(numerator / np.sqrt(xss * yss))
            cm = totals[..., 9 + v * 9:18 + v * 9].reshape(*totals.shape[:-1], 3, 3)
            denominators = cm.sum(axis=-1)
            if np.any(denominators == 0):
                raise ValueError('A resample lost a reference class')
            bas.append((np.diagonal(cm, axis1=-2, axis2=-1) / denominators).mean(axis=-1))
        return (np.stack(rs, axis=-1), np.stack(bas, axis=-1))
    r, ba = metrics(sufficient.sum(axis=0))
    for k in range(3):
        for v in range(2):
            np.testing.assert_allclose(r[k, v], stats.pearsonr(slopes[v], y[k]).statistic, atol=1e-12)
            cm = np.bincount(reference_classes[k] * 3 + classes[v], minlength=9).reshape(3, 3)
            np.testing.assert_allclose(ba[k, v], np.mean(np.diag(cm) / cm.sum(axis=1)), atol=1e-12)
    rng = np.random.default_rng(seed)
    dr = np.empty((reps, 3))
    dba = np.empty_like(dr)
    for start in range(0, reps, batch):
        stop = min(start + batch, reps)
        draws = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=stop - start)
        totals = (draws @ sufficient.reshape(n_blocks, -1)).reshape(stop - start, 3, 27)
        rr, bb = metrics(totals)
        dr[start:stop] = rr[:, :, 1] - rr[:, :, 0]
        dba[start:stop] = (bb[:, :, 1] - bb[:, :, 0]) * 100
    result = {'r': r.tolist(), 'ba_percent': (ba * 100).tolist(), 'delta_r': (r[:, 1] - r[:, 0]).tolist(), 'delta_ba_pp': ((ba[:, 1] - ba[:, 0]) * 100).tolist(), 'delta_r_ci95': np.quantile(dr, [0.025, 0.975], axis=0).T.tolist(), 'delta_ba_pp_ci95': np.quantile(dba, [0.025, 0.975], axis=0).T.tolist()}
    return (result, dr, dba)
