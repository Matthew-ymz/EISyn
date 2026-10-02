"""Analytic checks for the information-prior comparison (stdlib unittest)."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_shapley import coalition_ei_table, exact_shapley
from scripts.run_unicm_information_prior_comparison import (
    TOLERANCE_BITS, channel_priors, fit_prior, nonnegative, paired_bootstrap,
)
from scripts.run_unicm_synergy_regularized_calibration import CellDesign, predict_generalized_ridge


class InformationPriorTests(unittest.TestCase):
    def test_literature_priors_copy_and_cross_transfer(self):
        copy, native = channel_priors(np.eye(2), np.eye(2), ((0,), (1,)))
        np.testing.assert_allclose(copy["mim"], 0.5 * np.eye(2))
        np.testing.assert_allclose(copy["phi_si"], 0, atol=1e-12)
        np.testing.assert_allclose(copy["causal_density"], 0, atol=1e-12)
        cross, native = channel_priors(np.array([[0., 1.], [1., 0.]]), np.eye(2), ((0,), (1,)))
        np.testing.assert_allclose(native["conditional_te_edges"], [[0, 0.5], [0.5, 0]])
        np.testing.assert_allclose(cross["causal_density"], 0.25)
        np.testing.assert_allclose(native["phi_si_edges"], 1)

    def test_forward_si_includes_correlated_residuals(self):
        residual = np.array([[1., 0.5], [0.5, 1.]])
        _, native = channel_priors(np.eye(2), residual, ((0,), (1,)))
        expected = 0.5 * np.log2(1 / (1 - 0.5 ** 2))
        self.assertAlmostEqual(native["phi_si_edges"][0], expected)
        output_tc = 0.5 * np.log2(4 / (4 - 0.5 ** 2))
        self.assertAlmostEqual(native["phi_si_edges"][0] - native["phi_wms_edges"][0], output_tc)

    def test_cd_matches_joint_covariance_conditional_information(self):
        rng = np.random.default_rng(63)
        b = rng.normal(size=(6, 3))
        residual = np.diag([0.3, 0.7, 1.1])
        blocks = ((0, 3), (1, 4), (2, 5))
        priors, native = channel_priors(b, residual, blocks)
        full_joint = np.block([[np.eye(6), b], [b.T, residual + b.T @ b]])
        for source, block in enumerate(blocks):
            conditioned = [i for i in range(6) if i not in block]
            for target in range(3):
                if source == target:
                    self.assertEqual(native["conditional_te_edges"][source, target], 0)
                    continue
                indices = list(block) + [6 + target]
                conditional = full_joint[np.ix_(indices, indices)] - (
                    full_joint[np.ix_(indices, conditioned)] @ np.linalg.solve(
                        full_joint[np.ix_(conditioned, conditioned)],
                        full_joint[np.ix_(conditioned, indices)]))
                expected = 0.5 * (np.linalg.slogdet(conditional[:-1, :-1])[1]
                                  + np.log(conditional[-1, -1])
                                  - np.linalg.slogdet(conditional)[1]) / np.log(2)
                self.assertAlmostEqual(native["conditional_te_edges"][source, target], expected)
        self.assertAlmostEqual(priors["causal_density"][0].sum(), native["conditional_te_edges"].sum() / 6)

    def test_independent_copy_has_no_synergy_or_phi_r(self):
        priors, _ = channel_priors(np.eye(2), np.eye(2), ((0,), (1,)))
        for key in ("xi_shapley", "mmi_pid", "phi_r", "o_shapley", "phi_wms"):
            np.testing.assert_allclose(priors[key], 0, atol=1e-12)
        np.testing.assert_allclose(priors["ei_shapley"], 0.5 * np.eye(2))
        np.testing.assert_allclose(priors["conditional_mi"], 0.5 * np.eye(2))

    def test_cross_transfer_is_phi_r_without_source_synergy(self):
        coefficient = np.array([[0., 1.], [1., 0.]])
        priors, edges = channel_priors(coefficient, np.eye(2), ((0,), (1,)))
        np.testing.assert_allclose(edges["phi_r_edges"], [1.0])
        np.testing.assert_allclose(edges["phi_wms_edges"], [1.0])
        np.testing.assert_allclose(priors["phi_r"], 0.5)
        np.testing.assert_allclose(priors["phi_wms"], 0.5)
        np.testing.assert_allclose(priors["xi_shapley"], 0)
        np.testing.assert_allclose(priors["o_shapley"], 0)
        np.testing.assert_allclose(priors["mmi_pid"], 0)

    def test_scalar_sum_matches_analytic_mmi_and_conditional_mi(self):
        priors, edges = channel_priors(np.array([[1., 0.], [1., 1.]]), np.eye(2), ((0,), (1,)))
        self.assertAlmostEqual(edges["mmi_pid_edges"][0, 0], 0.5)
        self.assertAlmostEqual(priors["conditional_mi"][0, 1], 0.5)
        self.assertAlmostEqual(priors["conditional_mi"][0, 0], 0.5 * np.log2(3 / 2))
        self.assertAlmostEqual(priors["xi_shapley"][0].sum(), 0.5 * np.log2(4 / 3))

    def test_two_source_o_equals_xi_but_three_source_o_is_different(self):
        two, _ = channel_priors(np.array([[1., 0.], [1., 1.]]), np.eye(2), ((0,), (1,)))
        np.testing.assert_allclose(two["o_shapley"], two["xi_shapley"], atol=1e-12)
        three, _ = channel_priors(np.ones((3, 3)), np.eye(3), ((0,), (1,), (2,)))
        # Y=X1+X2+X3+noise: I(all;Y)=1 bit, I(any pair;Y)=0.5 bit.
        np.testing.assert_allclose(three["o_shapley"], np.full((3, 3), 1 / 6), atol=1e-12)
        np.testing.assert_allclose(three["xi_shapley"].sum(axis=1),
                                   1 - 1.5 * np.log2(4 / 3), atol=1e-12)
        self.assertGreater(np.max(np.abs(three["o_shapley"] - three["xi_shapley"])), 0.01)

    def test_o_shapley_matches_conditional_dtc_entropy_definition(self):
        rng = np.random.default_rng(24)
        coefficients = rng.normal(size=(6, 3))
        residual = np.diag([0.3, 0.7, 1.1])
        blocks = ((0, 3), (1, 4), (2, 5))
        priors, _ = channel_priors(coefficients, residual, blocks)
        for target in range(3):
            b = coefficients[:, target]
            conditional_covariance = np.eye(6) - np.outer(b, b) / (residual[target, target] + b @ b)
            game = {0: 0.0}
            for mask in range(1, 8):
                members = [m for m in range(3) if mask & (1 << m)]
                columns = [i for m in members for i in blocks[m]]
                joint_logdet = np.linalg.slogdet(conditional_covariance[np.ix_(columns, columns)])[1]
                omitted_sum = 0.0
                for member in members:
                    others = [i for m in members if m != member for i in blocks[m]]
                    if others:
                        omitted_sum += np.linalg.slogdet(conditional_covariance[np.ix_(others, others)])[1]
                game[mask] = 0.5 * (omitted_sum - (len(members) - 1) * joint_logdet) / np.log(2)
            np.testing.assert_allclose(priors["o_shapley"][target], exact_shapley(game, 3), atol=1e-12)

    def test_wms_joint_output_definition_and_phi_r_correction(self):
        # Two equal readouts, with independent unit noises: joint MI=0.5*log2(5).
        priors, edges = channel_priors(np.ones((2, 2)), np.eye(2), ((0,), (1,)))
        single_mi = 0.5 * np.log2(3 / 2)
        expected = 0.5 * np.log2(5) - 2 * single_mi
        self.assertAlmostEqual(edges["phi_wms_edges"][0], expected)
        self.assertAlmostEqual(edges["phi_r_edges"][0] - edges["phi_wms_edges"][0], single_mi)
        np.testing.assert_allclose(priors["phi_wms"], expected / 2)

    def test_vectorized_shapley_matches_reference_and_marginalization(self):
        rng = np.random.default_rng(12)
        coefficients = rng.normal(size=(6, 3))
        mixing = rng.normal(size=(3, 3))
        residual = mixing @ mixing.T + np.eye(3) * 0.2
        blocks = ((0, 3), (1, 4), (2, 5))
        audit = {}
        priors, edges = channel_priors(coefficients, residual, blocks, audit)
        for target in range(3):
            table = coalition_ei_table(coefficients[:, target:target + 1], residual[target:target + 1, target:target + 1], blocks)
            singles = [table[1 << m] for m in range(3)]
            game = {mask: value - sum(singles[m] for m in range(3) if mask & (1 << m))
                    for mask, value in table.items()}
            np.testing.assert_allclose(priors["xi_shapley"][target], exact_shapley(game, 3), atol=1e-12)
            np.testing.assert_allclose(priors["ei_shapley"][target], exact_shapley(table, 3), atol=1e-12)
        # Every native edge's mass must be allocated exactly once in total.
        np.testing.assert_allclose(priors["mmi_pid"].sum(axis=1), edges["mmi_pid_edges"].sum(axis=0))
        self.assertAlmostEqual(priors["phi_r"][0].sum(), edges["phi_r_edges"].sum())
        self.assertAlmostEqual(priors["phi_wms"][0].sum(), edges["phi_wms_edges"].sum())
        np.testing.assert_allclose(priors["phi_r"], np.broadcast_to(priors["phi_r"][0], (3, 3)))
        np.testing.assert_allclose(priors["xi_target_averaged"][0], priors["xi_shapley"].mean(axis=0))

    def test_tolerance_records_numerical_zero_and_rejects_violation(self):
        audit = {}
        result = nonnegative(np.array([-TOLERANCE_BITS / 2, 1.0]), "Syn", audit)
        np.testing.assert_array_equal(result, [0.0, 1.0])
        self.assertEqual(audit["Syn"]["numerical_zero_count"], 1)
        with self.assertRaisesRegex(ValueError, "minimum=.*threshold=.*affected_count=1"):
            nonnegative(np.array([-2 * TOLERANCE_BITS, 0.0]), "Syn", {})

    def test_shared_bootstrap_is_paired_and_recomputes_rmse(self):
        target = np.zeros((24, 2, 3))
        prediction = np.arange(24)[:, None, None] * np.ones_like(target) / 10
        aggregate, leads, counts = paired_bootstrap({"a": prediction, "b": prediction.copy()}, target,
                                                   np.ones((1, 2, 1)), 20, 6, 43)
        np.testing.assert_array_equal(aggregate["a"], aggregate["b"])
        np.testing.assert_array_equal(leads["a"], leads["b"])
        np.testing.assert_allclose(counts.sum(axis=1), 1)
        expected = np.sqrt(counts @ prediction[:, 0, 0] ** 2)
        np.testing.assert_allclose(aggregate["a"], expected)

    def test_overlapping_source_blocks_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            channel_priors(np.eye(2), np.eye(2), ((0,), (0,)))

    def test_fixed_mode_does_not_search_and_uses_exact_xi_settings(self):
        target = np.ones((3, 2, 24))
        prediction = target * 0.75
        with patch("scripts.run_unicm_information_prior_comparison.tune_prior") as search:
            with patch("scripts.run_unicm_information_prior_comparison.predict_generalized_ridge",
                       return_value=(prediction, prediction)) as predict:
                alpha, gamma, scores, _, _ = fit_prior({}, np.ones((2, 24, 2)), target,
                    np.ones((1, 2, 1)), alphas=[1, 2], gammas=[0, 1], floor_fraction=0.05,
                    fixed_parameters={"alpha": 30000.0, "gamma": 3.0})
        search.assert_not_called()
        self.assertEqual(alpha, 30000.0)
        self.assertEqual(gamma, 3.0)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(next(iter(scores.values())), 0.25)
        self.assertEqual(predict.call_args.kwargs, {"alpha": 30000.0, "gamma": 3.0, "floor_fraction": 0.05})

    def test_zero_gamma_predictions_equal_uniform_but_active_priors_can_differ(self):
        rng = np.random.default_rng(37)
        x_fit = rng.normal(size=(32, 44))
        y_fit = x_fit @ rng.normal(size=44)
        design = CellDesign(x_fit=x_fit, x_validation=rng.normal(size=(5, 44)),
                            x_test=rng.normal(size=(7, 44)), y_fit=y_fit, y_mean=0.,
                            gram=x_fit.T @ x_fit, rhs=x_fit.T @ y_fit)
        designs = {(target, lead): design for target in range(2) for lead in range(24)}
        priors = [np.broadcast_to(values, (2, 24, 11)) for values in
                  (np.ones(11), np.linspace(1, 5, 11), np.linspace(1, 5, 11) ** 2)]
        inactive = [predict_generalized_ridge(designs, prior, alpha=1000., gamma=0.,
                                             floor_fraction=.05) for prior in priors]
        for validation, test in inactive[1:]:
            np.testing.assert_array_equal(validation, inactive[0][0])
            np.testing.assert_array_equal(test, inactive[0][1])
        active = [predict_generalized_ridge(designs, prior, alpha=1000., gamma=3.,
                                           floor_fraction=.05)[1] for prior in priors]
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertGreater(np.max(np.abs(active[first] - active[second])), 1e-3)


if __name__ == "__main__":
    unittest.main()
