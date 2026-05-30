import importlib.util
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"

spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", MODULE_PATH)
dmf_fig6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmf_fig6
spec.loader.exec_module(dmf_fig6)


class LausanneZipLoadingTests(unittest.TestCase):
    def test_load_lausanne_zip_labels_and_drops_unknown_region(self) -> None:
        zip_path = ROOT / "data" / "Lausanne2008-33.zip"

        labels = dmf_fig6.load_lausanne_region_labels(zip_path)
        entries = dmf_fig6.load_lausanne_atlas_entries(zip_path)
        connectivity, kept_labels = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            return_labels=True,
        )

        self.assertEqual(connectivity.shape, (83, 83))
        self.assertEqual(len(kept_labels), 83)
        self.assertNotIn("Unknown", kept_labels)
        self.assertTrue(np.allclose(np.diag(connectivity), 0.0))
        self.assertTrue(np.isfinite(connectivity).all())
        raw_scaled = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            connectivity_scale=1.0,
        )
        self.assertTrue(np.allclose(connectivity, 0.2 * raw_scaled))

    def test_connectivity_scale_can_restore_raw_max_normalization(self) -> None:
        zip_path = ROOT / "data" / "Lausanne2008-33.zip"

        labels = dmf_fig6.load_lausanne_region_labels(zip_path)
        entries = dmf_fig6.load_lausanne_atlas_entries(zip_path)
        connectivity = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            connectivity_scale=1.0,
        )

        self.assertAlmostEqual(float(connectivity.max()), 0.5314685314685315)

    def test_pairwise_phi_metrics_match_manual_bivariate_formula(self) -> None:
        x0 = np.arange(1.0, 9.0)
        x1 = np.array([2.0, 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0])
        rates = np.column_stack([x0, x1])

        metrics = dmf_fig6.compute_pairwise_phi_metrics(rates)

        lagged = np.column_stack([rates[:-1, 0], rates[:-1, 1], rates[1:, 0], rates[1:, 1]])
        covariance = np.cov(lagged, rowvar=False, bias=False)
        tdmi = dmf_fig6.gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3])
        mi_self_0 = dmf_fig6.gaussian_mutual_information(covariance, sources=[0], targets=[2])
        mi_self_1 = dmf_fig6.gaussian_mutual_information(covariance, sources=[1], targets=[3])
        phi_wms = tdmi - mi_self_0 - mi_self_1
        redundancy = min(
            dmf_fig6.gaussian_mutual_information(covariance, sources=[source], targets=[target])
            for source in (0, 1)
            for target in (2, 3)
        )

        self.assertEqual(metrics["pair_count"], 1)
        self.assertAlmostEqual(metrics["phi_wms_mean"], phi_wms)
        self.assertAlmostEqual(metrics["phi_r_mean"], phi_wms + redundancy)

    def test_paper_hcp83_loader_requires_exact_hcp_lausanne83_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "hcp_lausanne83_connectivity.npy"
            with self.assertRaisesRegex(FileNotFoundError, "HCP Lausanne-83"):
                dmf_fig6.load_paper_hcp83_connectivity(missing_path)

            wrong_shape_path = Path(tmpdir) / "wrong.npy"
            np.save(wrong_shape_path, np.eye(82))
            with self.assertRaisesRegex(ValueError, "83x83"):
                dmf_fig6.load_paper_hcp83_connectivity(wrong_shape_path)

            valid_path = Path(tmpdir) / "valid.npy"
            valid = np.ones((83, 83), dtype=float)
            np.fill_diagonal(valid, 2.0)
            np.save(valid_path, valid)
            loaded = dmf_fig6.load_paper_hcp83_connectivity(valid_path)

            # The loader must not silently substitute local approximation files.
            self.assertEqual(loaded.shape, (83, 83))
            self.assertTrue(np.allclose(np.diag(loaded), 0.0))
            self.assertTrue(np.allclose(loaded, loaded.T))

    def test_paper_phi_r_metrics_use_phiid_atoms(self) -> None:
        time = np.linspace(0.0, 8.0 * np.pi, 240)
        bold = np.column_stack(
            [
                np.sin(time),
                np.sin(time + 0.25) + 0.1 * np.cos(2.0 * time),
            ]
        )

        metrics = dmf_fig6.compute_paper_phi_r_metrics(bold, tau=1)

        self.assertEqual(metrics["pair_count"], 1)
        self.assertEqual(metrics["phi_r_pairwise"].shape, (1,))
        self.assertTrue(np.isfinite(metrics["phi_r_mean"]))
        self.assertGreaterEqual(float(metrics["phi_r_mean"]), 0.0)
        self.assertIn("rtr", metrics["atom_keys"])
        self.assertEqual(metrics["phiid_redundancy"], "MMI")

    def test_paper_phi_r_smoke_run_writes_cache_and_metadata(self) -> None:
        connectivity = np.array(
            [
                [0.0, 0.15, 0.05],
                [0.15, 0.0, 0.08],
                [0.05, 0.08, 0.0],
            ],
            dtype=float,
        )
        parameters = dmf_fig6.DMFParameters(t_total=0.08, burn_in=0.02, dt=0.001, sigma=0.0)
        fic = dmf_fig6.FICParameters(max_iterations=2, tolerance_hz=50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "paper_phi_r.npz"
            figure_path = Path(tmpdir) / "paper_phi_r.png"
            result = dmf_fig6.reproduce_fig6b_paper_phi_r(
                connectivity=connectivity,
                g_values=[1.0, 1.1],
                parameters=parameters,
                fic_parameters=fic,
                expected_regions=3,
                max_regions=3,
                max_phi_pairs=2,
                results_path=results_path,
                figure_path=figure_path,
                seed=0,
            )

            self.assertTrue(results_path.exists())
            self.assertTrue(figure_path.exists())
            self.assertIn("phi_r_mean", result)
            self.assertIn("phi_r_pairwise", result)
            archive = np.load(results_path)
            for key in ("G", "mean_rate_hz", "bold_timeseries", "phi_r_mean", "phi_r_pairwise", "metadata"):
                self.assertIn(key, archive.files)
            self.assertGreaterEqual(float(np.nanmin(archive["phi_r_mean"])), 0.0)

    def test_whole_system_phi_eid_matches_conditional_total_correlation(self) -> None:
        transition = np.array(
            [
                [0.85, 0.35, 0.05],
                [0.10, 0.80, 0.30],
                [0.25, 0.20, 0.75],
            ],
            dtype=float,
        )
        noise_covariance = np.diag([0.20, 0.18, 0.22])

        metrics = dmf_fig6.compute_whole_system_phi_eid_from_gaussian_transition(
            transition,
            noise_covariance,
            source_covariance=np.eye(3),
        )

        self.assertGreaterEqual(float(metrics["phi_eid"]), -1.0e-9)
        self.assertAlmostEqual(
            float(metrics["phi_eid"]),
            float(metrics["conditional_total_correlation"]),
            places=9,
        )
        self.assertAlmostEqual(
            float(metrics["phi_eid"]),
            float(metrics["whole_ei"]) - float(np.sum(metrics["singleton_ei"])),
            places=9,
        )

    def test_phi_r_changes_with_sampling_distribution_but_mechanistic_phi_eid_does_not(self) -> None:
        rng = np.random.default_rng(7)
        transition = np.array([[0.75, 0.55], [0.20, 0.70]], dtype=float)
        noise_covariance = np.diag([0.08, 0.08])
        phi_eid = dmf_fig6.compute_whole_system_phi_eid_from_gaussian_transition(
            transition,
            noise_covariance,
            source_covariance=np.eye(2),
        )

        independent_source = rng.normal(size=(400, 2))
        correlated_latent = rng.normal(size=(400, 1))
        correlated_source = np.column_stack(
            [
                correlated_latent[:, 0] + 0.35 * rng.normal(size=400),
                correlated_latent[:, 0] + 0.35 * rng.normal(size=400),
            ]
        )
        independent_target = independent_source @ transition.T + rng.multivariate_normal(
            np.zeros(2),
            noise_covariance,
            size=400,
        )
        correlated_target = correlated_source @ transition.T + rng.multivariate_normal(
            np.zeros(2),
            noise_covariance,
            size=400,
        )

        independent_phi_r = dmf_fig6.compute_pairwise_phi_metrics_from_lagged_samples(
            independent_source,
            independent_target,
        )
        correlated_phi_r = dmf_fig6.compute_pairwise_phi_metrics_from_lagged_samples(
            correlated_source,
            correlated_target,
        )
        repeated_phi_eid = dmf_fig6.compute_whole_system_phi_eid_from_gaussian_transition(
            transition,
            noise_covariance,
            source_covariance=np.eye(2),
        )

        self.assertGreater(
            abs(float(independent_phi_r["phi_r_mean"]) - float(correlated_phi_r["phi_r_mean"])),
            0.02,
        )
        self.assertAlmostEqual(float(phi_eid["phi_eid"]), float(repeated_phi_eid["phi_eid"]), places=12)

    def test_average_pairwise_phi_eid_fallback_returns_finite_nonnegative_score(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(180, 4))
        transition = np.array(
            [
                [0.7, 0.4, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.3],
                [0.0, 0.0, 0.2, 0.7],
            ],
            dtype=float,
        )
        target = source @ transition.T + 0.25 * rng.normal(size=source.shape)

        metrics = dmf_fig6.compute_average_pairwise_phi_eid_from_lagged_samples(source, target)

        self.assertEqual(metrics["pair_count"], 6)
        self.assertTrue(np.isfinite(metrics["phi_eid_mean"]))
        self.assertGreaterEqual(float(metrics["phi_eid_mean"]), 0.0)

    def test_pairwise_phi_helpers_accept_max_pairs_for_fast_pilot_runs(self) -> None:
        rng = np.random.default_rng(17)
        source = rng.normal(size=(80, 5))
        target = 0.7 * source + 0.2 * rng.normal(size=source.shape)

        phi_r = dmf_fig6.compute_pairwise_phi_metrics_from_lagged_samples(source, target, max_pairs=3)
        phi_eid = dmf_fig6.compute_average_pairwise_phi_eid_from_lagged_samples(source, target, max_pairs=3)
        bootstrap = dmf_fig6.bootstrap_pairwise_phi_r(source, target, n_bootstrap=4, max_pairs=3)

        self.assertEqual(phi_r["pair_count"], 3)
        self.assertEqual(phi_eid["pair_count"], 3)
        self.assertEqual(bootstrap.shape, (4,))

    def test_iid_fig6_phi_eid_comparison_cli_smoke_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "reproduce_iid_fig6_phi_eid_comparison.py"),
                "--synthetic-smoke",
                "--figure",
                str(output_dir / "comparison.png"),
                "--results",
                str(output_dir / "comparison.npz"),
                "--doc",
                str(output_dir / "comparison.md"),
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            self.assertTrue((output_dir / "comparison.png").exists())
            self.assertGreater((output_dir / "comparison.png").stat().st_size, 1000)
            self.assertTrue((output_dir / "comparison.npz").exists())
            self.assertTrue((output_dir / "comparison.md").exists())
            archive = np.load(output_dir / "comparison.npz")
            for key in (
                "G",
                "mean_rate_hz",
                "phi_r",
                "phi_eid",
                "phi_r_bootstrap",
                "phi_r_variant_curves",
                "phi_r_variant_peak_g",
                "phi_r_variant_labels",
            ):
                self.assertIn(key, archive.files)
            self.assertGreaterEqual(float(np.nanmin(archive["phi_eid"])), -1.0e-9)
            self.assertTrue(np.isfinite(archive["phi_r_variant_peak_g"]).all())
            self.assertGreater(float(np.nanmin(archive["phi_r_variant_peak_g"])), 1.0)


if __name__ == "__main__":
    unittest.main()
