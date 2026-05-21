import json
import unittest
from pathlib import Path

import numpy as np

from transport_map_density import (
    AffineTransportMapDensityEstimator,
    fit_affine_transport_map_density,
    fit_quadratic_triangular_transport_map_density,
    multivariate_gaussian_logpdf,
    standard_gaussian_logpdf,
)


class TransportMapDensityTests(unittest.TestCase):
    def test_affine_transport_map_matches_gaussian_log_density(self) -> None:
        rng = np.random.default_rng(17)
        mean = np.array([0.8, -0.4], dtype=float)
        covariance = np.array([[1.6, 0.45], [0.45, 0.9]], dtype=float)
        train = rng.multivariate_normal(mean, covariance, size=6000)
        test = rng.multivariate_normal(mean, covariance, size=800)

        estimator = fit_affine_transport_map_density(train)

        self.assertIsInstance(estimator, AffineTransportMapDensityEstimator)
        log_hat = estimator.log_prob(test)
        log_true = multivariate_gaussian_logpdf(test, mean, covariance)
        rmse = float(np.sqrt(np.mean((log_hat - log_true) ** 2)))
        self.assertLess(rmse, 0.08)

    def test_pdf_matches_exponentiated_log_prob(self) -> None:
        rng = np.random.default_rng(23)
        samples = rng.normal(size=(200, 2))
        query = rng.normal(size=(20, 2))
        estimator = fit_affine_transport_map_density(samples)

        np.testing.assert_allclose(estimator.pdf(query), np.exp(estimator.log_prob(query)))

    def test_marginal_returns_working_density_estimator(self) -> None:
        rng = np.random.default_rng(31)
        samples = rng.normal(size=(300, 3))

        marginal = fit_affine_transport_map_density(samples).marginal([0])
        scores = marginal.log_prob(samples[:, [0]])

        self.assertEqual(marginal.dimension, 1)
        self.assertEqual(scores.shape, (samples.shape[0],))
        self.assertTrue(np.isfinite(scores).all())

    def test_quadratic_triangular_transport_map_matches_banana_density(self) -> None:
        rng = np.random.default_rng(41)
        beta = 0.42

        def banana_transform(base: np.ndarray) -> np.ndarray:
            warped = base.copy()
            warped[:, 1] = warped[:, 1] + beta * (warped[:, 0] ** 2 - 1.0)
            return warped

        def banana_inverse(samples: np.ndarray) -> np.ndarray:
            base = samples.copy()
            base[:, 1] = base[:, 1] - beta * (base[:, 0] ** 2 - 1.0)
            return base

        train = banana_transform(rng.normal(size=(5000, 2)))
        test = banana_transform(rng.normal(size=(1200, 2)))

        estimator = fit_quadratic_triangular_transport_map_density(train)
        log_hat = estimator.log_prob(test)
        log_true = standard_gaussian_logpdf(banana_inverse(test))
        rmse = float(np.sqrt(np.mean((log_hat - log_true) ** 2)))

        self.assertLess(rmse, 0.10)
        self.assertEqual(estimator.backend, "quadratic_triangular_transport_map")

    def test_demo_notebook_imports_tm_algorithm_only_from_standalone_module(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "transport_map_density_demo.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("from transport_map_density import", code_text)
        self.assertNotIn("from yrd", code_text)
        self.assertNotIn("import yrd", code_text)
        self.assertNotIn("from density_benchmark", code_text)
        self.assertNotIn("import density_benchmark", code_text)


if __name__ == "__main__":
    unittest.main()
