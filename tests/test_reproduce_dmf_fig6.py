import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.reproduce_dmf_fig6 import (
    Connectome,
    DMFParameters,
    FICParameters,
    calibrate_j_fic,
    compute_pairwise_phi_metrics,
    gaussian_mutual_information,
    load_brodmann_connectome,
    load_connectome,
    simulate_dmf,
)


ROOT = Path(__file__).resolve().parents[1]


class DmfFig6ReproductionTests(unittest.TestCase):
    def test_load_brodmann_connectome_validates_shape_and_zeros_diagonal(self) -> None:
        connectome = load_brodmann_connectome(ROOT / "data" / "connectome_brodmann82.npy")

        self.assertEqual(connectome.matrix.shape, (82, 82))
        self.assertEqual(len(connectome.labels), 82)
        self.assertTrue(np.isfinite(connectome.matrix).all())
        self.assertTrue(np.all(connectome.matrix >= 0.0))
        self.assertTrue(np.allclose(connectome.matrix, connectome.matrix.T))
        self.assertTrue(np.allclose(np.diag(connectome.matrix), 0.0))

    def test_load_connectome_accepts_generic_square_csv_and_max_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hcp.csv"
            np.savetxt(path, np.array([[0.0, 2.0], [4.0, 0.0]]), delimiter=",")

            connectome = load_connectome(path, normalize="max")

            self.assertIsInstance(connectome, Connectome)
            self.assertEqual(connectome.matrix.shape, (2, 2))
            self.assertTrue(np.allclose(connectome.matrix, connectome.matrix.T))
            self.assertAlmostEqual(float(connectome.matrix.max()), 1.0)
            self.assertTrue(np.allclose(np.diag(connectome.matrix), 0.0))

    def test_gaussian_mutual_information_distinguishes_independent_and_correlated(self) -> None:
        independent = np.eye(2)
        correlated = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=float)

        self.assertAlmostEqual(
            gaussian_mutual_information(independent, sources=[0], targets=[1]),
            0.0,
            places=9,
        )
        self.assertGreater(
            gaussian_mutual_information(correlated, sources=[0], targets=[1]),
            0.1,
        )

    def test_pairwise_phi_metrics_match_manual_bivariate_formula(self) -> None:
        x0 = np.arange(1.0, 9.0)
        x1 = np.array([2.0, 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0])
        rates = np.column_stack([x0, x1])
        metrics = compute_pairwise_phi_metrics(rates)

        lagged = np.column_stack([rates[:-1, 0], rates[:-1, 1], rates[1:, 0], rates[1:, 1]])
        covariance = np.cov(lagged, rowvar=False, bias=False)
        tdmi = gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3])
        mi_self_0 = gaussian_mutual_information(covariance, sources=[0], targets=[2])
        mi_self_1 = gaussian_mutual_information(covariance, sources=[1], targets=[3])
        phi_wms = tdmi - mi_self_0 - mi_self_1
        redundancy = min(
            gaussian_mutual_information(covariance, sources=[source], targets=[target])
            for source in (0, 1)
            for target in (2, 3)
        )

        self.assertEqual(metrics["pair_count"], 1)
        self.assertAlmostEqual(metrics["phi_wms_mean"], phi_wms, places=9)
        self.assertAlmostEqual(metrics["phi_r_mean"], phi_wms + redundancy, places=9)

    def test_cli_smoke_run_writes_cache_figure_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "reproduce_dmf_fig6.py"),
                "--connectome",
                str(ROOT / "data" / "connectome_brodmann82.npy"),
                "--output-dir",
                str(output_dir),
                "--g-count",
                "2",
                "--t-total",
                "0.012",
                "--burn-in",
                "0.004",
                "--dt",
                "0.001",
                "--j-fic-max-iters",
                "1",
                "--skip-trace",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            results_dir = output_dir / "results" / "dmf_fig6_brodmann82"
            figure_path = output_dir / "fig" / "dmf_fig6_brodmann82" / "fig6b_brodmann82_pilot.png"
            self.assertTrue((results_dir / "sweep.npz").exists())
            self.assertTrue((results_dir / "summary.csv").exists())
            self.assertTrue((results_dir / "metadata.json").exists())
            self.assertTrue(figure_path.exists())
            self.assertGreater(figure_path.stat().st_size, 1000)

            metadata = json.loads((results_dir / "metadata.json").read_text())
            self.assertEqual(metadata["connectome"]["n_regions"], 82)
            self.assertEqual(metadata["phi_metrics_source"], "excitatory_rate_gaussian_proxy")

    def test_fic_calibration_avoids_collapsed_zero_rate_solution(self) -> None:
        connectome = load_brodmann_connectome(ROOT / "data" / "connectome_brodmann82.npy")
        parameters = DMFParameters(t_total=0.06, burn_in=0.02, dt=0.001, sigma=0.0)
        fic = FICParameters(max_iterations=8)

        calibration = calibrate_j_fic(
            connectome.matrix,
            1.0,
            parameters=parameters,
            fic=fic,
            seed=0,
        )
        result = simulate_dmf(
            connectome.matrix,
            1.0,
            calibration["j_fic"],
            parameters=parameters,
            seed=0,
        )

        self.assertGreater(float(result["mean_rate_hz"]), 0.5)
        self.assertLess(abs(float(result["mean_rate_hz"]) - fic.target_rate_hz), 50.0)


if __name__ == "__main__":
    unittest.main()
