"""Analytic and distribution checks for the observational UniCM comparison."""
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss

from scripts.unicm_observational_information import (
    GaussianObservation, all_observed_coalitions, assign_surd_specific,
    fit_observation, gaussian_surd_allocation, nonnegative_ridge_scores,
    observed_cd_information, observed_pair_information, observational_prior,
)


class ObservationalInformationTests(unittest.TestCase):
    def test_surd_xor_redundancy_and_unique_cases(self):
        for table, r_expected, s_expected in (
            ([0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]),
            ([0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]),
        ):
            r, s = assign_surd_specific(np.asarray(table), 2)
            np.testing.assert_allclose(r, r_expected)
            np.testing.assert_allclose(s, s_expected)
        table = np.asarray([float(mask & 6 == 6) for mask in range(8)])
        _, synergy = assign_surd_specific(table, 3)
        self.assertEqual(synergy[6], 1)
        self.assertEqual(synergy.sum(), 1)

    def test_gaussian_shortcut_matches_statewise_quadrature(self):
        rng = np.random.default_rng(72)
        a = rng.normal(size=(3, 3))
        cxx = a @ a.T + np.eye(3)
        b = np.asarray([0.7, -0.2, 0.4])
        cross = cxx @ b[:, None]
        cyy = np.asarray([[float(b @ cxx @ b) + 0.4]])
        density = GaussianObservation(cxx, cross, cyy, ((0,), (1,), (2,)), 3, 1)
        table = all_observed_coalitions(density)
        allocation = gaussian_surd_allocation(table, 3, {})
        nodes, weights = hermgauss(32)
        r, s = np.zeros(8), np.zeros(8)
        rho2 = -np.expm1(-2 * np.log(2) * table[:, 0])
        for node, weight in zip(nodes * np.sqrt(2), weights / np.sqrt(np.pi)):
            specific = table[:, 0] + rho2 * (node ** 2 - 1) / (2 * np.log(2))
            ri, si = assign_surd_specific(specific, 3)
            r += weight * ri
            s += weight * si
        np.testing.assert_allclose(allocation['redundancy_unique_atoms'][:, 0], r, atol=1e-12)
        np.testing.assert_allclose(allocation['synergy_atoms'][:, 0], s, atol=1e-12)
        self.assertLess(allocation['closure_error_bit'], 1e-12)

    def test_natural_source_correlation_is_preserved_and_wms_is_signed(self):
        cxx = np.asarray([[1., .9], [.9, 1.]])
        density = GaussianObservation(cxx, cxx.copy(), cxx + np.eye(2), ((0,), (1,)), 2, 1)
        wms = observed_pair_information(density, 'phi_wms')['edges'][0, 0]
        self.assertAlmostEqual(wms, .5 * np.log2(4 - .9 ** 2) - 1)
        self.assertLess(wms, 0)
        si = observed_pair_information(density, 'phi_si')
        self.assertAlmostEqual(si['edges'][0, 0], 0)
        self.assertLess(si['si_identity_error_bit'], 1e-12)
        transfer = observed_cd_information(density)['transfer']
        np.testing.assert_allclose(transfer, 0, atol=1e-12)
        cross_density = GaussianObservation(cxx, cxx[:, ::-1], cxx + np.eye(2), ((0,), (1,)), 2, 1)
        cross_transfer = observed_cd_information(cross_density)['transfer']
        self.assertAlmostEqual(cross_transfer[0, 0, 1], .5 * np.log2(2 - .9 ** 2))
        self.assertEqual(cross_transfer[0, 0, 0], 0)

    def test_signed_score_mapping_preserves_all_pairwise_differences(self):
        raw = np.asarray([[[-.3, .1, -.1]]])
        transformed, record = nonnegative_ridge_scores(raw, 'phi_wms')
        np.testing.assert_allclose(transformed, [[[0, .4, .2]]])
        np.testing.assert_allclose(np.diff(transformed), np.diff(raw))
        self.assertEqual(record['negative_raw_count'], 2)
        self.assertAlmostEqual(record['maximum_common_offset_bit'], .3)

    def test_source_covariance_depends_on_observations_not_independence(self):
        rng = np.random.default_rng(32)
        h = rng.normal(size=(80, 2, 2))
        h[:, 1] = h[:, 0] + .01 * rng.normal(size=(80, 2))
        y = rng.normal(size=(80, 2, 1))
        density = fit_observation(h, y)
        self.assertGreater(density.source_covariance[0, 1], .99)

    def test_all_methods_use_actual_target_and_have_audited_timing(self):
        rng = np.random.default_rng(18)
        h = rng.normal(size=(200, 3, 2))
        y = .4 * h[:, :, :1] + rng.normal(size=(200, 3, 1))
        for method in ('phi_r', 'phi_wms', 'phi_si', 'causal_density', 'surd'):
            prior, native, audit = observational_prior(h, y, method)
            self.assertEqual(prior.shape, (3, 1, 3))
            self.assertTrue(np.isfinite(prior).all())
            self.assertTrue(np.all(prior >= 0))
            self.assertGreater(audit['timing']['core_total_seconds'], 0)
            self.assertIn('raw_attribution', native)
        _, first, _ = observational_prior(h, y, 'surd')
        _, second, _ = observational_prior(h, y[rng.permutation(len(y))], 'surd')
        self.assertGreater(np.max(np.abs(first['coalition_mi'] - second['coalition_mi'])), .01)


if __name__ == '__main__':
    unittest.main()
