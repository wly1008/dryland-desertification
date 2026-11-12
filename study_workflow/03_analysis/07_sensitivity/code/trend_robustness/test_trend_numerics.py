import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next((p for p in _ArchivePath(__file__).resolve().parents if (p / '07_environment/project_paths.py').is_file()))
_archive_sys.path.insert(0, str(_archive_root / '07_environment'))
from project_paths import PROJECT_ROOT, CODE_ROOT, parameter_file, FIGURES_ROOT, TABLES_ROOT, remap_path, figure_output_path
import itertools
import unittest
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from trend_numeric import sen_mk, bh_adjust, states

class TrendTests(unittest.TestCase):

    def test_sen_against_scipy(self):
        rng = np.random.default_rng(41)
        years = np.arange(2001, 2022)
        values = rng.normal(size=(21, 20))
        values[:, 0] = np.arange(21)
        values[:, 1] = np.round(values[:, 1])
        slope, *_ = sen_mk(values, years)
        for i in range(values.shape[1]):
            self.assertAlmostEqual(slope[i], stats.theilslopes(values[:, i], years).slope, places=13)

    def test_mk_tie_variance_by_exact_permutations(self):
        permutations = np.array(sorted(set(itertools.permutations([0, 0, 1, 2, 2]))), dtype=float).T
        _, p, s, variance = sen_mk(permutations, np.arange(5))
        reference = []
        for values in permutations.T:
            reference.append(sum((np.sign(values[j] - values[i]) for i in range(5) for j in range(i + 1, 5))))
        np.testing.assert_array_equal(s, reference)
        np.testing.assert_allclose(variance, np.var(reference), atol=1e-12)
        expected = 2 * stats.norm.sf(np.abs(np.array(reference) - np.sign(reference)) / np.sqrt(np.var(reference)))
        np.testing.assert_allclose(p, expected, atol=1e-14)

    def test_constants_and_tie_dominated_zero_sen(self):
        values = np.stack([np.ones(21), np.r_[np.zeros(6), np.ones(15)]], axis=1)
        slope, p, s, v = sen_mk(values, np.arange(21))
        self.assertEqual(p[0], 1)
        self.assertEqual(v[0], 0)
        self.assertEqual(slope[1], 0)
        self.assertLess(p[1], 0.05)
        np.testing.assert_array_equal(states(slope, p), [1, 1])

    def test_bh_against_statsmodels(self):
        p = np.r_[0, 1, 0.001, 0.001, 0.05, np.random.default_rng(17).uniform(size=1000)]
        expected = multipletests(p, method='fdr_bh')[1]
        np.testing.assert_allclose(bh_adjust(p), expected, rtol=0, atol=1e-15)
if __name__ == '__main__':
    unittest.main(verbosity=2)
