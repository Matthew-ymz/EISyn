import unittest

import numpy as np
import pandas as pd

from exp.tm_ei_l_baseline_support import (
    build_baseline_target,
    corrected_ei_table,
    fit_log_l_slope,
    raw_and_baseline_curve_table,
    resolve_l_sample_count,
    select_best_corrected_baselines,
)


class TmEiLBaselineSupportTests(unittest.TestCase):
    def test_resolve_l_sample_count_scales_with_intervention_volume(self) -> None:
        self.assertEqual(
            resolve_l_sample_count(
                l_value=8.0,
                source_dim=2,
                sample_count_mode="fixed_density",
                reference_l=4.0,
                reference_n_samples=4096,
                min_n_samples=128,
            ),
            16384,
        )
        self.assertEqual(
            resolve_l_sample_count(
                l_value=1.0,
                source_dim=2,
                sample_count_mode="fixed_density",
                reference_l=4.0,
                reference_n_samples=4096,
                min_n_samples=128,
            ),
            256,
        )

    def test_resolve_l_sample_count_keeps_fixed_mode_constant(self) -> None:
        self.assertEqual(
            resolve_l_sample_count(
                l_value=16.0,
                source_dim=2,
                sample_count_mode="fixed",
                reference_l=4.0,
                reference_n_samples=4096,
                min_n_samples=128,
            ),
            4096,
        )

    def test_fit_log_l_slope_recovers_known_log_l_coefficient(self) -> None:
        frame = pd.DataFrame(
            {
                "L": [1.0, 2.0, 4.0, 8.0],
                "ei": [0.25 + 1.7 * np.log(value) for value in [1.0, 2.0, 4.0, 8.0]],
            }
        )

        summary = fit_log_l_slope(frame, value_column="ei")

        self.assertAlmostEqual(summary["slope_log_l"], 1.7, places=10)
        self.assertAlmostEqual(summary["intercept"], 0.25, places=10)

    def test_corrected_ei_table_subtracts_matching_baseline_rows(self) -> None:
        runs = pd.DataFrame(
            [
                {"dynamics": "identity", "L": 1.0, "seed": 0, "ei_tm": 2.0},
                {"dynamics": "identity", "L": 2.0, "seed": 0, "ei_tm": 3.0},
            ]
        )
        baselines = pd.DataFrame(
            [
                {"dynamics": "identity", "baseline": "id", "L": 1.0, "seed": 0, "baseline_ei_tm": 1.5},
                {"dynamics": "identity", "baseline": "id", "L": 2.0, "seed": 0, "baseline_ei_tm": 2.8},
            ]
        )

        corrected = corrected_ei_table(runs, baselines)

        np.testing.assert_allclose(corrected["ei_corrected"].to_numpy(), np.array([0.5, 0.2]))
        self.assertEqual(list(corrected["baseline"]), ["id", "id"])

    def test_gain_matched_baseline_learns_linear_gain(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.uniform(-2.0, 2.0, size=(256, 1))
        target = 2.5 * source

        baseline = build_baseline_target(
            baseline="gain_matched",
            source=source,
            target_signal=target,
            noise_std=0.0,
            rng=np.random.default_rng(11),
        )

        np.testing.assert_allclose(baseline.target, target, atol=1e-10)
        self.assertAlmostEqual(baseline.metadata["gain_matrix"][0][0], 2.5, places=10)

    def test_select_best_corrected_baselines_picks_flattest_curve(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "label": "identity",
                    "noise_mode": "fixed",
                    "baseline": "identity",
                    "corrected_slope_log_l": 0.2,
                    "corrected_range_l": 0.1,
                },
                {
                    "label": "identity",
                    "noise_mode": "fixed",
                    "baseline": "analytic_volume",
                    "corrected_slope_log_l": -0.01,
                    "corrected_range_l": 0.05,
                },
                {
                    "label": "product",
                    "noise_mode": "fixed",
                    "baseline": "gain_matched",
                    "corrected_slope_log_l": 0.04,
                    "corrected_range_l": 0.07,
                },
            ]
        )

        best = select_best_corrected_baselines(summary, noise_mode="fixed")

        self.assertEqual(best.loc["identity", "baseline"], "analytic_volume")
        self.assertEqual(best.loc["product", "baseline"], "gain_matched")

    def test_select_best_corrected_baselines_can_filter_large_curved_residuals(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "label": "tanh",
                    "noise_mode": "fixed",
                    "baseline": "shuffled_target",
                    "corrected_slope_log_l": 0.01,
                    "corrected_range_l": 1.4,
                },
                {
                    "label": "identity",
                    "noise_mode": "fixed",
                    "baseline": "variance_matched",
                    "corrected_slope_log_l": 0.02,
                    "corrected_range_l": 0.08,
                },
            ]
        )

        best = select_best_corrected_baselines(
            summary,
            noise_mode="fixed",
            max_corrected_range=0.2,
        )

        self.assertIn("identity", best.index)
        self.assertNotIn("tanh", best.index)

    def test_raw_and_baseline_curve_table_contains_raw_and_all_baseline_series(self) -> None:
        runs = pd.DataFrame(
            [
                {"label": "identity", "noise_mode": "fixed", "L": 1.0, "seed": 0, "ei_tm": 2.0},
                {"label": "identity", "noise_mode": "fixed", "L": 2.0, "seed": 0, "ei_tm": 3.0},
            ]
        )
        baselines = pd.DataFrame(
            [
                {
                    "label": "identity",
                    "noise_mode": "fixed",
                    "baseline": "identity",
                    "L": 1.0,
                    "seed": 0,
                    "baseline_ei_tm": 1.5,
                },
                {
                    "label": "identity",
                    "noise_mode": "fixed",
                    "baseline": "identity",
                    "L": 2.0,
                    "seed": 0,
                    "baseline_ei_tm": 2.7,
                },
            ]
        )

        curves = raw_and_baseline_curve_table(runs, baselines, noise_mode="fixed")

        self.assertEqual(set(curves["series"]), {"raw EI", "identity"})
        raw_curve = curves[curves["series"].eq("raw EI")].sort_values("L")
        baseline_curve = curves[curves["series"].eq("identity")].sort_values("L")
        np.testing.assert_allclose(raw_curve["ei_mean"].to_numpy(), np.array([2.0, 3.0]))
        np.testing.assert_allclose(baseline_curve["ei_mean"].to_numpy(), np.array([1.5, 2.7]))


if __name__ == "__main__":
    unittest.main()
