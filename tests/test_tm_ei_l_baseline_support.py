import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from utils import (
    build_baseline_target,
    corrected_ei_table,
    fit_log_l_slope,
    plot_raw_ei_by_center,
    raw_and_baseline_curve_table,
    resolve_l_sample_count,
    run_ei_center_sweep,
    sample_uniform_source,
    select_best_corrected_baselines,
    select_top_global_baselines,
    simulate_known_dynamics,
    plot_raw_vs_selected_baselines_by_l,
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

    def test_sample_uniform_source_uses_zero_centered_intervention_range(self) -> None:
        source = sample_uniform_source(
            l_value=6.0,
            n_samples=20000,
            source_dim=1,
            rng=np.random.default_rng(5),
        )

        self.assertGreaterEqual(float(source.min()), -3.0)
        self.assertLessEqual(float(source.max()), 3.0)
        self.assertAlmostEqual(float(source.mean()), 0.0, delta=0.04)

    def test_sample_uniform_source_can_use_nonzero_center(self) -> None:
        source = sample_uniform_source(
            l_value=4.0,
            center=3.0,
            n_samples=20000,
            source_dim=1,
            rng=np.random.default_rng(13),
        )

        self.assertGreaterEqual(float(source.min()), 1.0)
        self.assertLessEqual(float(source.max()), 5.0)
        self.assertAlmostEqual(float(source.mean()), 3.0, delta=0.04)

    def test_requested_one_dimensional_dynamics_match_formulas(self) -> None:
        source = np.array([[-2.0], [-0.5], [0.0], [1.5]])

        for dynamics, expected in [
            ("identity", source),
            ("square", source**2),
            ("sine", np.sin(source)),
            ("exponential", np.exp(source)),
        ]:
            signal, target = simulate_known_dynamics(
                dynamics,
                source,
                noise_std=0.0,
                rng=np.random.default_rng(11),
            )
            np.testing.assert_allclose(signal, expected)
            np.testing.assert_allclose(target, expected)

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

    def test_select_top_global_baselines_excludes_negative_controls_and_breaks_ties(self) -> None:
        summary = pd.DataFrame(
            [
                {"label": "x", "noise_mode": "fixed", "baseline": "null", "corrected_slope_log_l": 0.0},
                {"label": "x^2", "noise_mode": "fixed", "baseline": "null", "corrected_slope_log_l": 0.0},
                {"label": "x", "noise_mode": "fixed", "baseline": "shuffled_target", "corrected_slope_log_l": 0.0},
                {"label": "x^2", "noise_mode": "fixed", "baseline": "shuffled_target", "corrected_slope_log_l": 0.0},
                {"label": "x", "noise_mode": "fixed", "baseline": "variance_matched", "corrected_slope_log_l": 0.10},
                {"label": "x^2", "noise_mode": "fixed", "baseline": "variance_matched", "corrected_slope_log_l": -0.20},
                {"label": "x", "noise_mode": "fixed", "baseline": "identity", "corrected_slope_log_l": 0.10},
                {"label": "x^2", "noise_mode": "fixed", "baseline": "identity", "corrected_slope_log_l": 0.30},
                {"label": "x", "noise_mode": "fixed", "baseline": "gain_matched", "corrected_slope_log_l": 0.20},
                {"label": "x^2", "noise_mode": "fixed", "baseline": "gain_matched", "corrected_slope_log_l": 0.20},
                {"label": "x", "noise_mode": "scaled_with_l", "baseline": "analytic_volume", "corrected_slope_log_l": 0.01},
                {"label": "x^2", "noise_mode": "scaled_with_l", "baseline": "analytic_volume", "corrected_slope_log_l": 0.01},
            ]
        )

        ranking = select_top_global_baselines(summary, noise_mode="fixed", top_n=2)

        self.assertEqual(list(ranking["baseline"]), ["variance_matched", "identity"])
        np.testing.assert_allclose(ranking["mean_abs_corrected_slope"].to_numpy(), np.array([0.15, 0.20]))
        np.testing.assert_allclose(ranking["median_abs_corrected_slope"].to_numpy(), np.array([0.15, 0.20]))

    def test_plot_raw_vs_selected_baselines_by_l_uses_baseline_ei(self) -> None:
        runs = pd.DataFrame(
            [
                {
                    "label": label,
                    "noise_mode": "fixed",
                    "L": l_value,
                    "ei_tm": 10.0 + float(l_value),
                }
                for label in ["x", "x^2"]
                for l_value in [1.0, 2.0]
            ]
        )
        baselines = pd.DataFrame(
            [
                {
                    "label": label,
                    "noise_mode": "fixed",
                    "baseline": baseline,
                    "L": l_value,
                    "baseline_ei_tm": float(l_value) if baseline == "identity" else 2.0 * float(l_value),
                }
                for label in ["x", "x^2"]
                for baseline in ["identity", "variance_matched"]
                for l_value in [1.0, 2.0]
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            png_path, pdf_path, curve_table = plot_raw_vs_selected_baselines_by_l(
                runs,
                baselines,
                output_dir=Path(tmp_dir),
                noise_mode="fixed",
                dynamics_labels=("x", "x^2"),
                baselines=("identity", "variance_matched"),
                output_stem="focused_top2",
            )

            self.assertEqual(png_path.name, "focused_top2.png")
            self.assertEqual(pdf_path.name, "focused_top2.pdf")
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            x_identity = curve_table[
                curve_table["label"].eq("x") & curve_table["series"].eq("identity")
            ].sort_values("L")
            np.testing.assert_allclose(x_identity["ei_mean"].to_numpy(), np.array([1.0, 2.0]))

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

    def test_raw_and_baseline_curve_table_preserves_requested_label_order(self) -> None:
        runs = pd.DataFrame(
            [
                {"label": "sin(x)", "noise_mode": "fixed", "L": 1.0, "seed": 0, "ei_tm": 2.0},
                {"label": "x", "noise_mode": "fixed", "L": 1.0, "seed": 0, "ei_tm": 1.0},
                {"label": "exp(x)", "noise_mode": "fixed", "L": 1.0, "seed": 0, "ei_tm": 3.0},
            ]
        )
        baselines = pd.DataFrame(
            [
                {"label": "sin(x)", "noise_mode": "fixed", "baseline": "identity", "L": 1.0, "seed": 0, "baseline_ei_tm": 1.5},
                {"label": "x", "noise_mode": "fixed", "baseline": "identity", "L": 1.0, "seed": 0, "baseline_ei_tm": 0.5},
                {"label": "exp(x)", "noise_mode": "fixed", "baseline": "identity", "L": 1.0, "seed": 0, "baseline_ei_tm": 2.5},
            ]
        )

        curves = raw_and_baseline_curve_table(
            runs,
            baselines,
            noise_mode="fixed",
            dynamics_labels=("x", "sin(x)", "exp(x)"),
        )

        self.assertEqual(list(dict.fromkeys(curves["label"])), ["x", "sin(x)", "exp(x)"])

    def test_run_ei_center_sweep_returns_one_row_per_case_center_and_repeat(self) -> None:
        results = run_ei_center_sweep(
            center_values=(-1.0, 0.0, 1.0),
            l_value=4.0,
            dynamics_specs=(
                {"dynamics": "identity", "source_dim": 1, "target_dim": 1, "label": "x"},
                {"dynamics": "sine", "source_dim": 1, "target_dim": 1, "label": "sin(x)"},
            ),
            baselines=("identity", "variance_matched"),
            n_samples=64,
            repeats=2,
            base_noise_std=0.0,
            seed=23,
        )
        runs = results["runs"]
        baselines = results["baselines"]

        self.assertEqual(len(runs), 12)
        self.assertEqual(len(baselines), 24)
        self.assertEqual(set(runs["center"]), {-1.0, 0.0, 1.0})
        self.assertEqual(set(baselines["center"]), {-1.0, 0.0, 1.0})
        self.assertEqual(set(runs["L"]), {4.0})
        self.assertEqual(set(runs["label"]), {"x", "sin(x)"})
        self.assertEqual(set(baselines["baseline"]), {"identity", "variance_matched"})
        self.assertTrue(np.isfinite(runs["ei_tm"]).all())
        self.assertTrue(np.isfinite(baselines["baseline_ei_tm"]).all())

    def test_plot_raw_ei_by_center_writes_raw_and_baseline_curves_without_std_band(self) -> None:
        runs = pd.DataFrame(
            [
                {"label": label, "center": center, "ei_tm": value}
                for label, base in [("sin(x)", 3.0), ("x", 1.0), ("exp(x)", 5.0)]
                for center, value in [(-1.0, base), (0.0, base + 0.5), (1.0, base + 1.0)]
            ]
        )
        baselines = pd.DataFrame(
            [
                {
                    "label": label,
                    "center": center,
                    "baseline": baseline,
                    "baseline_ei_tm": value,
                }
                for label, base in [("sin(x)", 3.0), ("x", 1.0), ("exp(x)", 5.0)]
                for baseline, offset in [("identity", 0.1), ("variance_matched", 0.2)]
                for center, value in [(-1.0, base + offset), (0.0, base + 0.5 + offset), (1.0, base + 1.0 + offset)]
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            png_path, pdf_path, curve_table = plot_raw_ei_by_center(
                runs,
                baselines,
                output_dir=Path(tmp_dir),
                dynamics_labels=("x", "sin(x)", "exp(x)"),
                baselines=("identity", "variance_matched"),
                output_stem="center_curves",
            )

            self.assertEqual(png_path.name, "center_curves.png")
            self.assertEqual(pdf_path.name, "center_curves.pdf")
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertEqual(list(dict.fromkeys(curve_table["label"])), ["x", "sin(x)", "exp(x)"])
            self.assertEqual(set(curve_table["series"]), {"raw EI", "identity", "variance_matched"})
            self.assertNotIn("ei_std", curve_table.columns)


if __name__ == "__main__":
    unittest.main()
