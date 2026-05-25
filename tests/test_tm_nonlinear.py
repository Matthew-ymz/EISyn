import unittest

import numpy as np
import pandas as pd

from scripts.tm_nonlinear import (
    build_shap_peid_comparison,
    run_alpha_sweep_tm,
    run_beta_sweep_tm,
    signed_real_power,
    summarize_tm_runs,
)


class TmNonlinearSupportTests(unittest.TestCase):
    def test_alpha_sweep_summary_contains_expected_ratios(self) -> None:
        runs = run_alpha_sweep_tm(
            alpha_values=(0.0, 1.0),
            n_samples=160,
            repeats=2,
            L=2.0,
            q1_noise_std=0.05,
            seed=7,
        )

        summary = summarize_tm_runs(runs, group_columns=["alpha", "L", "q1_noise_std"])

        self.assertEqual(summary.shape[0], 2)
        self.assertTrue(np.isfinite(summary["tm_syn_ratio"]).all())
        self.assertTrue(np.isfinite(summary["tm_single_q2_ratio"]).all())
        self.assertTrue(np.isfinite(summary["tm_single_q3_ratio"]).all())

    def test_beta_sweep_keeps_fractional_powers_real_on_symmetric_support(self) -> None:
        values = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        transformed = signed_real_power(values, 0.5)

        self.assertTrue(np.isfinite(transformed).all())
        self.assertAlmostEqual(float(transformed[0]), -2.0)
        self.assertAlmostEqual(float(transformed[-1]), 2.0)
        np.testing.assert_allclose(signed_real_power(values, 0.0), np.ones_like(values))

    def test_beta_sweep_summary_contains_expected_ratios(self) -> None:
        runs = run_beta_sweep_tm(
            beta_values=(0.0, 0.5, 1.0),
            n_samples=160,
            repeats=2,
            L=2.0,
            q1_noise_std=0.05,
            seed=7,
        )

        summary = summarize_tm_runs(runs, group_columns=["beta", "L", "q1_noise_std"])

        self.assertEqual(summary.shape[0], 3)
        self.assertTrue(np.isfinite(summary["tm_syn_ratio"]).all())
        self.assertTrue(np.isfinite(summary["tm_single_q2_ratio"]).all())
        self.assertTrue(np.isfinite(summary["tm_single_q3_ratio"]).all())

    def test_build_shap_peid_comparison_merges_matching_settings(self) -> None:
        tm_summary = pd.DataFrame(
            [
                {
                    "alpha": 0.0,
                    "L": 2.0,
                    "q1_noise_std": 0.05,
                    "tm_ei_mean": 1.0,
                    "tm_syn_mean": 0.0,
                    "tm_syn_ratio": 0.0,
                    "tm_single_q2_ratio": 1.0,
                    "tm_single_q3_ratio": 0.0,
                }
            ]
        )
        shap_summary = pd.DataFrame(
            [
                {
                    "alpha": 0.0,
                    "L": 2.0,
                    "q1_noise_std": 0.05,
                    "shap_interaction_share_mean": 0.01,
                    "model_r2_mean": 0.99,
                }
            ]
        )

        comparison = build_shap_peid_comparison(tm_summary, shap_summary)

        self.assertEqual(comparison.shape[0], 1)
        self.assertAlmostEqual(float(comparison.loc[0, "tm_syn_ratio"]), 0.0)
        self.assertAlmostEqual(float(comparison.loc[0, "shap_interaction_share_mean"]), 0.01)


if __name__ == "__main__":
    unittest.main()
