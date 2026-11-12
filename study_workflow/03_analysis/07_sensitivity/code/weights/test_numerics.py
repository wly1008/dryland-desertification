"""Small numerical reference tests; the production run also audits real baselines."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import unittest
import numpy as np
from scipy import stats
from common import entropy_weights, ols, paired_external, trend_classes
import config as c

class NumericalChecks(unittest.TestCase):

    def test_ols_against_scipy(self):
        rng = np.random.default_rng(7)
        years = np.arange(2001, 2022)
        values = rng.normal(size=(21, 8)) + (years - years.mean())[:, None] * np.linspace(-0.6, 0.6, 8)
        slopes, p = ols(values, years)
        for i in range(8):
            fit = stats.linregress(years, values[:, i])
            self.assertAlmostEqual(slopes[i], fit.slope, places=13)
            self.assertAlmostEqual(p[i], fit.pvalue, places=13)

    def test_constant_and_exact_linear(self):
        years = np.arange(2001, 2022)
        values = np.stack([np.full(21, 0.5), np.arange(21), -np.arange(21)], axis=1)
        slope, p = ols(values, years)
        np.testing.assert_array_equal(trend_classes(slope, p), [1, 0, 2])
        self.assertEqual(p[0], 1)

    def test_entropy_equivalence_and_npv_renormalization(self):
        counts = np.array([[12, 28, 60], [20, 40, 40], [70, 20, 10]], dtype=float)
        _, w = entropy_weights(counts)
        q = counts / counts.sum(axis=1)[:, None]
        er = (q * np.log2(counts)).sum(axis=1) / np.log2(counts.sum(axis=1))
        legacy = (1 - er) / (1 - er).sum()
        np.testing.assert_allclose(w, legacy, atol=1e-14)
        _, removed = entropy_weights(counts[:-1])
        np.testing.assert_allclose(removed, w[:-1] / (1 - w[-1]), atol=1e-14)

    def test_paired_metrics_and_deterministic_draws(self):
        rng = np.random.default_rng(17)
        nblocks, perblock = (40, 12)
        n = nblocks * perblock
        x = rng.normal(size=(2, n)) + np.array([[2], [-1]])
        y = rng.normal(size=(3, n)) + 3
        classes = rng.integers(0, 3, size=(2, n), dtype=np.uint8)
        reference = np.tile(np.arange(3, dtype=np.uint8), n // 3)
        references = np.stack([reference, np.roll(reference, 1), np.roll(reference, 2)])
        blocks = np.repeat(np.arange(nblocks), perblock)
        old_replicates = c.BOOTSTRAP_REPLICATES
        try:
            c.BOOTSTRAP_REPLICATES = 23
            first = paired_external(x, classes, y, references, blocks, nblocks)
            second = paired_external(x, classes, y, references, blocks, nblocks)
        finally:
            c.BOOTSTRAP_REPLICATES = old_replicates
        self.assertEqual(first, second)
        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(first['r'][i][j], stats.pearsonr(x[j], y[i]).statistic, places=12)
if __name__ == '__main__':
    unittest.main(verbosity=2)
